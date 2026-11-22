//! Immutable VARCHAR pages for the live SQL scan.
use super::{db, Connection};
use std::{
    collections::{hash_map::Entry, HashMap},
    ffi::c_void,
    ptr, slice,
};

// DuckDB keeps strings of up to twelve bytes inside the vector slot.
type InlineText = db::duckdb_string_t__bindgen_ty_1__bindgen_ty_2;
const _: () = assert!(size_of::<InlineText>() == size_of::<db::duckdb_string_t>());

#[derive(Default)]
pub(super) struct TextColumn {
    slots: Vec<InlineText>,
    long: Vec<(usize, String)>,
    pages: Vec<Option<DictionaryPage>>,
}

impl TextColumn {
    pub(super) fn push(&mut self, text: &str) {
        let mut slot = InlineText {
            length: 0,
            inlined: [0; 12],
        };
        if text.len() <= slot.inlined.len() {
            slot.length = text.len() as u32;
            for (dst, src) in slot.inlined.iter_mut().zip(text.bytes()) {
                *dst = src as _;
            }
        } else {
            self.long.push((self.slots.len(), String::from(text)));
        }
        self.slots.push(slot);
    }

    pub(super) fn append(&mut self, tail: Self) {
        let offset = self.slots.len();
        self.slots.extend(tail.slots);
        self.long
            .extend(tail.long.into_iter().map(|(row, s)| (offset + row, s)));
    }

    pub(super) fn can_extend_idle(&self, additional: usize, size: usize, limit: usize) -> bool {
        self.slots.len().saturating_add(additional) <= self.slots.capacity()
            && (self.pages.len() < limit || self.slots.len().saturating_add(additional) / size <= self.pages.capacity())
    }

    pub(super) fn idle_charge(&self, additional: usize, size: usize) -> usize {
        use super::scan::vector_idle_charge;
        let new_pages = self.slots.len().saturating_add(additional) / size - self.pages.len();
        // Allow for label offsets, backing text, indices, validity, and the
        // native vector. This is an allowance, not DuckDB allocator accounting.
        const PAGE_BYTES: usize = 64 * 1024;
        vector_idle_charge(&self.slots, additional)
            .saturating_add(vector_idle_charge(&self.pages, new_pages))
            .saturating_add(new_pages.saturating_mul(PAGE_BYTES))
    }

    pub(super) fn encode_pages(&mut self, conn: &Connection, nulls: &[usize]) {
        let size = unsafe { db::duckdb_vector_size() } as usize;
        // Build pages only once, after all their rows are present. A rewrite
        // publishes a new column; the old snapshot keeps its immutable pages.
        for page in self.pages.len()..self.slots.len() / size {
            let encoded = self
                .dictionary(page * size, size, nulls)
                .and_then(|data| data.into_page(conn));
            self.pages.push(encoded);
        }
    }

    fn dictionary(&self, start: usize, count: usize, nulls: &[usize]) -> Option<Dictionary> {
        // Prefer flat storage if there is little repetition. A page is only
        // worth encoding when each label represents several rows on average.
        let max_labels = count / 8;
        let mut labels = HashMap::new();
        let mut data = Dictionary {
            indices: Vec::with_capacity(count),
            offsets: vec![0],
            bytes: Vec::new(),
            validity: Vec::new(),
            null_count: 0,
        };
        let mut long = self.long[self.long.partition_point(|(row, _)| *row < start)..]
            .iter()
            .peekable();
        let mut nulls = nulls[nulls.partition_point(|&row| row < start)..]
            .iter()
            .copied()
            .peekable();
        for row in start..start + count {
            if nulls.next_if_eq(&row).is_some() {
                data.indices.push(0);
                if data.validity.is_empty() {
                    data.validity.resize(count.div_ceil(8), 0xff);
                }
                let i = row - start;
                data.validity[i / 8] &= !(1 << (i % 8));
                data.null_count += 1;
                continue;
            }
            let text = if let Some((_, text)) = long.next_if(|(index, _)| *index == row) {
                text.as_bytes()
            } else {
                let slot = &self.slots[row];
                // The inline slots were populated with exactly length bytes.
                unsafe { slice::from_raw_parts(slot.inlined.as_ptr().cast(), slot.length as usize) }
            };
            let label = labels.len();
            let id = match labels.entry(text) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    if label == max_labels {
                        return None;
                    }
                    let end = i32::try_from(data.bytes.len().checked_add(text.len())?).ok()?;
                    data.bytes.extend_from_slice(text);
                    data.offsets.push(end);
                    *entry.insert(label as i32)
                }
            };
            data.indices.push(id);
        }
        (!labels.is_empty()).then_some(data)
    }

    pub(super) unsafe fn reference_page(&self, output: db::duckdb_vector, index: usize) -> bool {
        if let Some(Some(page)) = self.pages.get(index) {
            db::duckdb_vector_reference_vector(output, db::duckdb_data_chunk_get_vector(page.0, 0));
            return true;
        }
        false
    }

    pub(super) unsafe fn copy_flat(&self, output: db::duckdb_vector, start: usize, end: usize) {
        ptr::copy_nonoverlapping(
            self.slots.as_ptr().add(start),
            db::duckdb_vector_get_data(output).cast(),
            end - start,
        );
        let first = self.long.partition_point(|(row, _)| *row < start);
        for (row, text) in self.long[first..].iter().take_while(|(row, _)| *row < end) {
            db::duckdb_vector_assign_string_element_len(
                output,
                (*row - start) as u64,
                text.as_ptr().cast(),
                text.len() as u64,
            );
        }
    }
}

struct DictionaryPage(db::duckdb_data_chunk);

// Each chunk is fully built before publication. Scan threads only read the
// source vector's metadata and acquire a DuckDB-owned reference to its
// immutable buffer. DuckDB writes to the separately owned output vector.
// The snapshot keeps this source chunk alive until the last scan drops it.
unsafe impl Send for DictionaryPage {}
unsafe impl Sync for DictionaryPage {}

impl Drop for DictionaryPage {
    fn drop(&mut self) {
        unsafe { db::duckdb_destroy_data_chunk(&mut self.0) };
    }
}

struct Dictionary {
    indices: Vec<i32>,
    offsets: Vec<i32>,
    bytes: Vec<u8>,
    validity: Vec<u8>,
    null_count: i64,
}

// Arrow's parent array keeps both child descriptors and the backing data in
// stable storage until DuckDB invokes its release callback.
struct ArrowPage {
    _data: Dictionary,
    dictionary: db::ArrowArray,
    dictionary_buffers: [*const c_void; 3],
    indices: db::ArrowArray,
    index_buffers: [*const c_void; 2],
    root_buffers: [*const c_void; 1],
    children: [*mut db::ArrowArray; 1],
}

impl Dictionary {
    fn into_page(mut self, conn: &Connection) -> Option<DictionaryPage> {
        let schema = arrow_schema(conn)?;
        let len = self.indices.len();
        let labels = self.offsets.len() - 1;
        let null_count = self.null_count;
        if self.bytes.is_empty() {
            self.bytes.push(0); // the Arrow data buffer also exists for empty strings
        }
        let page = Box::new(ArrowPage {
            dictionary: db::ArrowArray::empty(),
            dictionary_buffers: [
                ptr::null(),
                self.offsets.as_ptr().cast(),
                self.bytes.as_ptr().cast(),
            ],
            indices: db::ArrowArray::empty(),
            index_buffers: [
                if self.validity.is_empty() {
                    ptr::null()
                } else {
                    self.validity.as_ptr().cast()
                },
                self.indices.as_ptr().cast(),
            ],
            root_buffers: [ptr::null()],
            children: [ptr::null_mut()],
            _data: self,
        });
        let page = Box::into_raw(page);
        let mut chunk = ptr::null_mut();
        unsafe {
            // Form the self-references after committing the heap allocation
            // to Arrow. The consumer owns the allocation through private_data.
            (*page).dictionary.length = labels as i64;
            (*page).dictionary.n_buffers = 3;
            (*page).dictionary.buffers = ptr::addr_of_mut!((*page).dictionary_buffers).cast();
            (*page).dictionary.release = Some(release_child);
            (*page).indices.length = len as i64;
            (*page).indices.null_count = null_count;
            (*page).indices.n_buffers = 2;
            (*page).indices.buffers = ptr::addr_of_mut!((*page).index_buffers).cast();
            (*page).indices.dictionary = ptr::addr_of_mut!((*page).dictionary);
            (*page).indices.release = Some(release_child);
            (*page).children[0] = ptr::addr_of_mut!((*page).indices);
            let mut root = db::ArrowArray::empty();
            root.length = len as i64;
            root.n_buffers = 1;
            root.buffers = ptr::addr_of_mut!((*page).root_buffers).cast();
            root.n_children = 1;
            root.children = ptr::addr_of_mut!((*page).children).cast();
            root.release = Some(release_page);
            root.private_data = page.cast();
            let mut error =
                db::duckdb_data_chunk_from_arrow(conn.conn, &mut root, schema.0, &mut chunk);
            let failed = db::duckdb_error_data_has_error(error);
            db::duckdb_destroy_error_data(&mut error);
            if failed {
                db::duckdb_destroy_data_chunk(&mut chunk);
            }
            // DuckDB clears the release pointer when it takes ownership.
            if let Some(release) = root.release {
                release(&mut root);
            }
            if failed {
                return None;
            }
        }
        (!chunk.is_null()).then(|| DictionaryPage(chunk))
    }
}

unsafe extern "C" fn release_child(child: *mut db::ArrowArray) {
    (*child).release = None;
}

unsafe extern "C" fn release_page(root: *mut db::ArrowArray) {
    (*root).release = None;
    drop(Box::from_raw((*root).private_data.cast::<ArrowPage>()));
}

struct ArrowSchema(db::duckdb_arrow_converted_schema);

impl Drop for ArrowSchema {
    fn drop(&mut self) {
        unsafe { db::duckdb_destroy_arrow_converted_schema(&mut self.0) };
    }
}

fn arrow_schema(conn: &Connection) -> Option<ArrowSchema> {
    unsafe extern "C" fn release(schema: *mut db::ArrowSchema) {
        (*schema).release = None;
    }
    let mut labels = db::ArrowSchema::empty();
    labels.format = c"u".as_ptr();
    labels.name = c"".as_ptr();
    labels.release = Some(release);
    let mut indices = db::ArrowSchema::empty();
    indices.format = c"i".as_ptr();
    indices.name = c"text".as_ptr();
    indices.flags = 2; // ARROW_FLAG_NULLABLE
    indices.dictionary = &mut labels;
    indices.release = Some(release);
    let mut child = &mut indices as *mut db::ArrowSchema;
    let mut root = db::ArrowSchema::empty();
    root.format = c"+s".as_ptr();
    root.name = c"".as_ptr();
    root.n_children = 1;
    root.children = &mut child;
    root.release = Some(release);
    let mut compiled = ptr::null_mut();
    unsafe {
        let mut error = db::duckdb_schema_from_arrow(conn.conn, &mut root, &mut compiled);
        let failed = db::duckdb_error_data_has_error(error);
        db::duckdb_destroy_error_data(&mut error);
        release(&mut root);
        if failed {
            db::duckdb_destroy_arrow_converted_schema(&mut compiled);
            return None;
        }
    }
    (!compiled.is_null()).then(|| ArrowSchema(compiled))
}
