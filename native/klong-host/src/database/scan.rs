//! A SQL view over a live Klong table. The interpreter publishes an immutable,
//! thread-safe snapshot before a query. DuckDB workers never access a Klong Rc.
use super::{cstring, db, text::TextColumn, Connection};
use core::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicUsize, Ordering},
};
use klong_core::{error::Result, table::Table, Error, Interp, Value};
use std::{boxed::Box, format, sync::Arc, vec::Vec};
use std::{ffi::CString, sync::Mutex};

pub(super) struct Relation {
    value: Value,
    revision: u64,
    defer_idle: bool,
    shared: Shared,
}
type Shared = Arc<Mutex<Published>>;

struct Published {
    snapshot: Arc<Snapshot>,
    bind_error: Option<CString>,
}

struct Snapshot {
    rows: usize,
    columns: Vec<(CString, Column)>,
}

struct Column {
    data: Data,
    nulls: Vec<usize>,
}

enum Data {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Text(TextColumn),
}

// Charge each new logical item. If append would grow its Vec, instead
// charge the entire replacement allocation conservatively. Earlier charged
// allocations stay counted, including when realloc temporarily holds both.
pub(super) fn vector_idle_charge<T>(items: &Vec<T>, additional: usize) -> usize {
    let needed = items.len().saturating_add(additional);
    let elements = if needed <= items.capacity() { additional } else { needed.max(8).saturating_mul(2) };
    elements.saturating_mul(core::mem::size_of::<T>())
}

impl Column {
    fn from_value(ip: &Interp, v: &Value, start: usize, end: usize, previous_type: Option<db::duckdb_type>) -> Result<Self> {
        let mut nulls = Vec::new();
        let data = match v {
            Value::Ints(a) if a.rank() == 1 => Data::Int(a.data[start..end].iter().map(|&x| x as i64).collect()),
            Value::Flts(a) if a.rank() == 1 => Data::Float(a.data[start..end].iter().map(|&x| x as f64).collect()),
            Value::List(list) => {
                let items = &list[start..end];
                let ty = match items.iter().find(|v| !matches!(v, Value::Undefined)) {
                    Some(Value::Str(_) | Value::Char(_) | Value::Sym(_)) => db::DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR,
                    Some(Value::Flt(_)) => db::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE,
                    Some(Value::Int(_)) => db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT,
                    None => previous_type.unwrap_or(db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT),
                    _ => return Err(Error::typ("unsupported SQL column value")),
                };
                let mut result = match ty {
                    db::DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR => Data::Text(TextColumn::default()),
                    db::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE => Data::Float(Vec::with_capacity(items.len())),
                    _ => Data::Int(Vec::with_capacity(items.len())),
                };
                for (i, item) in items.iter().enumerate() {
                    if matches!(item, Value::Undefined) {
                        nulls.push(start + i);
                    }
                    match (&mut result, item) {
                        (Data::Text(col), Value::Str(s)) => col.push(s.as_str()),
                        (Data::Text(col), Value::Char(c)) => col.push(c.encode_utf8(&mut [0; 4])),
                        (Data::Text(col), Value::Sym(s)) => col.push(ip.syms.name(*s)),
                        (Data::Text(col), Value::Undefined) => col.push(""),
                        (Data::Int(col), Value::Int(i)) => col.push(*i as i64),
                        (Data::Int(col), Value::Undefined) => col.push(0),
                        (Data::Float(col), Value::Flt(x)) => col.push(*x as f64),
                        (Data::Float(col), Value::Undefined) => col.push(0.0),
                        _ => return Err(Error::typ("SQL column has incompatible cell types")),
                    }
                }
                result
            }
            _ => return Err(Error::typ("expected a one-dimensional SQL column")),
        };
        Ok(Self { data, nulls })
    }

    // The caller has validated every column before appending any of them.
    fn append(&mut self, other: Column, conn: &Connection) {
        self.nulls.extend(other.nulls);
        match (&mut self.data, other.data) {
            (Data::Int(a), Data::Int(b)) => a.extend(b),
            (Data::Float(a), Data::Float(b)) => a.extend(b),
            (Data::Text(a), Data::Text(b)) => {
                a.append(b);
                a.encode_pages(conn, &self.nulls);
            }
            _ => unreachable!("validated column types"),
        }
    }

    fn can_extend_idle(&self, additional: usize, size: usize) -> bool {
        // Vec growth can copy the complete old allocation. Large growth belongs
        // on the normal query path, where the requested data is needed.
        const MAX_MOVE: usize = 32_768;
        if self.nulls.len() >= MAX_MOVE && self.nulls.len().saturating_add(additional) > self.nulls.capacity() {
            return false;
        }
        match &self.data {
            Data::Int(xs) => xs.len() < MAX_MOVE || xs.len().saturating_add(additional) <= xs.capacity(),
            Data::Float(xs) => xs.len() < MAX_MOVE || xs.len().saturating_add(additional) <= xs.capacity(),
            Data::Text(xs) => xs.can_extend_idle(additional, size, MAX_MOVE),
        }
    }

    fn idle_charge(&self, additional: usize, size: usize) -> usize {
        let cells = match &self.data {
            Data::Int(xs) => vector_idle_charge(xs, additional),
            Data::Float(xs) => vector_idle_charge(xs, additional),
            Data::Text(xs) => xs.idle_charge(additional, size),
        };
        cells.saturating_add(vector_idle_charge(&self.nulls, additional))
    }

    fn sql_type(&self) -> db::duckdb_type {
        match self.data {
            Data::Int(_) => db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT,
            Data::Float(_) => db::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE,
            Data::Text(_) => db::DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR,
        }
    }
    unsafe fn copy(&self, vec: db::duckdb_vector, start: usize, end: usize, size: usize) {
        let count = end - start;
        match &self.data {
            Data::Int(v) => ptr::copy_nonoverlapping(v.as_ptr().add(start), db::duckdb_vector_get_data(vec).cast(), count),
            Data::Float(v) => ptr::copy_nonoverlapping(v.as_ptr().add(start), db::duckdb_vector_get_data(vec).cast(), count),
            Data::Text(v) => {
                if count == size && v.reference_page(vec, start / size) {
                    return;
                }
                v.copy_flat(vec, start, end);
            }
        }
        // DuckDB can reuse a vector previously filled with nulls. Obtain a
        // writable mask before restoring validity for this batch.
        let first = self.nulls.partition_point(|&row| row < start);
        let last = self.nulls.partition_point(|&row| row < end);
        if first < last || !db::duckdb_vector_get_validity(vec).is_null() {
            db::duckdb_vector_ensure_validity_writable(vec);
            let valid = db::duckdb_vector_get_validity(vec);
            ptr::write_bytes(valid, 0xff, count.div_ceil(64));
            for row in &self.nulls[first..last] {
                db::duckdb_validity_set_row_invalid(valid, (*row - start) as u64);
            }
        }
    }
}

fn snapshot(conn: &Connection, v: &Value, ip: &Interp) -> Result<(u64, Arc<Snapshot>)> {
    let t = Table::from_value(v).ok_or_else(|| Error::typ(".db expects table values"))?.data();
    let mut columns = Vec::new();
    for (name, value) in t.columns() {
        let Value::Str(name) = name else {
            return Err(Error::typ("invalid column name"));
        };
        let mut col = Column::from_value(ip, value, 0, t.row_count(), None)?;
        if let Data::Text(text) = &mut col.data {
            text.encode_pages(conn, &col.nulls);
        }
        columns.push((cstring(name.as_str())?, col));
    }
    Ok((t.revision(), Arc::new(Snapshot { rows: t.row_count(), columns })))
}

impl Relation {
    pub(super) fn register(conn: &Connection, name: &str, value: &Value, ip: &Interp) -> Result<Self> {
        let (revision, data) = snapshot(conn, value, ip)?;
        let shared = Arc::new(Mutex::new(Published { snapshot: data, bind_error: None }));
        // Every .db(dictionary) has a private catalog. The quoted function and view
        // live in distinct namespaces while the generated name avoids user input in a call.
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let scan_name = format!("kg_table_scan_{}", NEXT.fetch_add(1, Ordering::Relaxed));
        let c_name = cstring(&scan_name)?;
        unsafe {
            let mut function = db::duckdb_create_table_function();
            if function.is_null() {
                return Err(Error::memory());
            }
            db::duckdb_table_function_set_name(function, c_name.as_ptr());
            db::duckdb_table_function_set_extra_info(function, Box::into_raw(Box::new(shared.clone())).cast(), Some(free::<Shared>));
            db::duckdb_table_function_set_bind(function, Some(bind));
            db::duckdb_table_function_set_init(function, Some(init));
            db::duckdb_table_function_set_function(function, Some(scan));
            let status = db::duckdb_register_table_function(conn.conn, function);
            db::duckdb_destroy_table_function(&mut function);
            if status != db::duckdb_state_DuckDBSuccess {
                return Err(Error::system("could not register SQL table"));
            }
        }
        let quoted = name.replace('"', "\"\"");
        conn.query(&format!("CREATE TEMP VIEW \"{quoted}\" AS SELECT * FROM {scan_name}()"))?;
        Ok(Self { value: value.clone(), revision, defer_idle: false, shared })
    }
    /// Prepare at most one complete vector's worth of rows. Incompatible
    /// tails, rewrites and wide schemas use the ordinary foreground refresh.
    pub(super) fn idle_append(&mut self, conn: &Connection, ip: &Interp, budget: usize) -> Option<(usize, usize, usize)> {
        let table = Table::from_value(&self.value)?.data();
        if self.defer_idle || table.revision() == self.revision || table.last_rewrite_revision() > self.revision
            || table.columns().len() > 8
        {
            return None;
        }
        let size = unsafe { db::duckdb_vector_size() } as usize;
        let mut published = self.shared.lock().ok()?;
        let start = published.snapshot.rows;
        let end = (start / size + 1).checked_mul(size)?;
        if end > table.row_count() || published.bind_error.is_some() || Arc::strong_count(&published.snapshot) != 1 {
            return None;
        }
        if start >= 32_768 && !published.snapshot.columns.iter().all(|(_, col)| col.can_extend_idle(end - start, size)) {
            self.defer_idle = true;
            return None;
        }
        let charge = published.snapshot.columns.iter().fold(0usize, |sum, (_, col)| {
            sum.saturating_add(col.idle_charge(end - start, size))
        });
        if charge > budget {
            return None;
        }
        // Bound bytes as well as rows. The foreground has the full VARCHAR
        // path, including arbitrary length values.
        let small = table.columns().iter().all(|(_, v)| match v {
            Value::Ints(a) => a.rank() == 1,
            Value::Flts(a) => a.rank() == 1,
            Value::List(values) => values[start..end].iter().all(|v| match v {
                Value::Undefined | Value::Int(_) | Value::Flt(_) | Value::Char(_) => true,
                Value::Str(s) => s.as_str().len() <= 12,
                Value::Sym(s) => ip.syms.name(*s).len() <= 12,
                _ => false,
            }),
            _ => false,
        });
        if !small || !published.append_suffix(conn, ip, &table, end).unwrap_or(false) {
            self.defer_idle = true;
            return None;
        }
        if end == table.row_count() {
            self.revision = table.revision();
        }
        Some((start, end, charge))
    }

    pub(super) fn refresh(&mut self, conn: &Connection, ip: &Interp) -> Result<()> {
        let table = Table::from_value(&self.value).ok_or_else(|| Error::typ("expected a live table"))?;
        let data = table.data();
        if data.revision() == self.revision {
            return Ok(());
        }
        let mut guard = self.shared.lock().map_err(|_| Error::system("SQL table snapshot lock failed"))?;
        // A malformed live view must only fail statements that bind it. In
        // particular, ROLLBACK must reach DuckDB even if a Klong column changed.
        let append = guard.bind_error.is_none() && data.last_rewrite_revision() <= self.revision;
        let result = guard.update(conn, &self.value, ip, append);
        self.defer_idle = result.is_err();
        guard.bind_error = match result {
            Ok(()) => None,
            Err(error) => Some(cstring(&error.message)?),
        };
        self.revision = data.revision();
        Ok(())
    }
}

impl Published {
    fn update(&mut self, conn: &Connection, value: &Value, ip: &Interp, append: bool) -> Result<()> {
        let table = Table::from_value(value).ok_or_else(|| Error::typ("expected a live table"))?.data();
        if append && self.append_suffix(conn, ip, &table, table.row_count())? {
            return Ok(());
        }
        self.snapshot = snapshot(conn, value, ip)?.1;
        Ok(())
    }

    fn append_suffix(
        &mut self,
        conn: &Connection,
        ip: &Interp,
        table: &klong_core::table::TableData,
        end: usize,
    ) -> Result<bool> {
        let Some(current) = Arc::get_mut(&mut self.snapshot) else { return Ok(false) };
        if table.columns().len() != current.columns.len() {
            return Ok(false);
        }
        let suffix = table.columns().iter().zip(&current.columns)
            .map(|((_, v), (_, old))| Column::from_value(ip, v, current.rows, end, Some(old.sql_type())))
            .collect::<Result<Vec<_>>>()?;
        if suffix.iter().zip(&current.columns).any(|(a, (_, b))| a.sql_type() != b.sql_type()) {
            return Ok(false);
        }
        for ((_, col), tail) in current.columns.iter_mut().zip(suffix) {
            col.append(tail, conn);
        }
        current.rows = end;
        Ok(true)
    }
}

// DuckDB takes ownership of one Box at each set_*_data call and releases it
// with its corresponding monomorphized callback, including on query failure.
unsafe extern "C" fn free<T>(p: *mut c_void) {
    drop(Box::from_raw(p.cast::<T>()));
}

unsafe extern "C" fn bind(info: db::duckdb_bind_info) {
    let shared = &*db::duckdb_bind_get_extra_info(info).cast::<Shared>();
    let Ok(published) = shared.lock() else {
        db::duckdb_bind_set_error(info, c"SQL table snapshot unavailable".as_ptr());
        return;
    };
    if let Some(error) = &published.bind_error {
        db::duckdb_bind_set_error(info, error.as_ptr());
        return;
    }
    let snapshot = &published.snapshot;
    for (name, col) in &snapshot.columns {
        let mut ty = db::duckdb_create_logical_type(col.sql_type());
        db::duckdb_bind_add_result_column(info, name.as_ptr(), ty);
        db::duckdb_destroy_logical_type(&mut ty);
    }
    db::duckdb_bind_set_cardinality(info, snapshot.rows as u64, true);
    db::duckdb_bind_set_bind_data(info, Box::into_raw(Box::new(snapshot.clone())).cast(), Some(free::<Arc<Snapshot>>));
}

unsafe extern "C" fn init(info: db::duckdb_init_info) {
    db::duckdb_init_set_init_data(info, Box::into_raw(Box::new(AtomicUsize::new(0))).cast(), Some(free::<AtomicUsize>));
}

unsafe extern "C" fn scan(info: db::duckdb_function_info, chunk: db::duckdb_data_chunk) {
    let data = &*db::duckdb_function_get_bind_data(info).cast::<Arc<Snapshot>>();
    let offset = &*db::duckdb_function_get_init_data(info).cast::<AtomicUsize>();
    let size = db::duckdb_vector_size() as usize;
    let start = offset.fetch_add(size, Ordering::Relaxed).min(data.rows);
    let end = start.saturating_add(size).min(data.rows);
    if end > start {
        for (i, (_, col)) in data.columns.iter().enumerate() {
            col.copy(db::duckdb_data_chunk_get_vector(chunk, i as u64), start, end, size);
        }
    }
    db::duckdb_data_chunk_set_size(chunk, (end - start) as u64);
}
