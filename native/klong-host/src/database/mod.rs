//! Desktop SQL. Each callable owns its connection and any live Klong relations.
mod scan;
mod text;

use std::{
    collections::HashMap, format, rc::{Rc, Weak}, time::{Duration, Instant}, vec::Vec,
};
use core::{cell::RefCell, ptr};
use klong_core::{error::Result, value::Func, Error, Flt, Int, Interp, Value};
use libduckdb_sys as db;
use std::ffi::{CStr, CString};

struct LiveRelation {
    relation: RefCell<scan::Relation>,
    conn: Rc<Connection>,
    idle: Weak<IdlePublication>,
}

impl Drop for LiveRelation {
    fn drop(&mut self) {
        if let Some(idle) = self.idle.upgrade() {
            idle.schedule.borrow_mut().remove(self as *const Self);
        }
    }
}

pub(crate) fn register(ip: &mut Interp, idle: Option<Rc<IdlePublication>>) {
    ip.register(
        ".db",
        1,
        Rc::new(move |ip, args| {
            let (path, tables) = match args.first().ok_or_else(|| Error::arity(".db needs an argument"))? {
                Value::Str(s) => (s.as_str(), None),
                Value::Dict(d) => (":memory:", Some(d.borrow())),
                _ => return Err(Error::typ(".db expects a path or a dictionary of tables")),
            };
            let conn = Rc::new(Connection::open(path)?);
            let mut relations = Vec::new();
            let mut captures = Vec::new();
            if let Some(d) = tables {
                for (key, value) in d.keys().iter().zip(d.vals()) {
                    let Value::Str(name) = key else {
                        return Err(Error::typ("database names must be strings"));
                    };
                    let relation = scan::Relation::register(&conn, name.as_str(), value, ip)?;
                    let idle = idle.as_ref().map(Rc::downgrade).unwrap_or_default();
                    relations.push(Rc::new(LiveRelation { relation: RefCell::new(relation), conn: conn.clone(), idle }));
                    captures.push(value.clone());
                }
            }
            if let Some(idle) = &idle {
                idle.add(&relations);
            }
            let query = Rc::new(move |ip: &mut Interp, args: &[Value]| {
                let Value::Str(sql) = args.first().ok_or_else(|| Error::arity("database query needs an argument"))? else {
                    return Err(Error::typ("SQL query must be a string"));
                };
                for live in &relations {
                    live.relation.borrow_mut().refresh(&conn, ip)?;
                    if let Some(idle) = live.idle.upgrade() {
                        idle.schedule.borrow_mut().release(Rc::as_ptr(live));
                    }
                }
                conn.query(sql.as_str())?.values()
            });
            Ok(Value::Fn(Rc::new(Func::native_with_captures(ip.syms.intern(".db.query"), 1, query, captures))))
        }),
    );
}

/// The CLI alone calls this scheduler, between requests on the interpreter thread.
#[derive(Default)]
pub(crate) struct IdlePublication {
    schedule: RefCell<Schedule>,
    trace: bool,
}

// A registered relation's Rc allocation has a stable address. Drop removes
// this key before the allocation can be reused. The ring never dereferences it.
type RelationId = *const LiveRelation;

// A single relation may prepare pages ahead of its next ordinary refresh.
const IDLE_BYTES: usize = 8 * 1024 * 1024;
// A page converts at most eight short columns before committing them. Keep
// headroom for suffix vectors, dictionary builders, and transient metadata.
const PAGE_HEADROOM: usize = 1024 * 1024;

struct Lease {
    owner: RelationId,
    charged: usize,
}

struct Entry {
    relation: Weak<LiveRelation>,
    previous: RelationId,
    next: RelationId,
}

#[derive(Default)]
struct Schedule {
    relations: HashMap<RelationId, Entry>,
    head: Option<RelationId>,
    remaining: usize,
    quiet_since: Option<Instant>,
    lease: Option<Lease>,
}

impl Schedule {
    fn insert(&mut self, relation: Weak<LiveRelation>) {
        let key = relation.as_ptr();
        let entry = match self.head {
            Some(head) => {
                let tail = self.relations[&head].previous;
                self.relations.get_mut(&tail).unwrap().next = key;
                self.relations.get_mut(&head).unwrap().previous = key;
                Entry { relation, previous: tail, next: head }
            }
            None => {
                self.head = Some(key);
                Entry { relation, previous: key, next: key }
            }
        };
        self.relations.insert(key, entry);
    }

    fn remove(&mut self, key: RelationId) {
        let Some(entry) = self.relations.remove(&key) else { return };
        if self.relations.is_empty() {
            self.head = None;
            self.relations = HashMap::new();
        } else {
            self.relations.get_mut(&entry.previous).unwrap().next = entry.next;
            self.relations.get_mut(&entry.next).unwrap().previous = entry.previous;
            if self.head == Some(key) {
                self.head = Some(entry.next);
            }
        }
        self.remaining = self.remaining.min(self.relations.len());
        self.release(key);
    }

    fn release(&mut self, key: RelationId) {
        if self.lease.as_ref().is_some_and(|lease| lease.owner == key) {
            self.lease = None;
            self.remaining = self.relations.len();
        }
    }

    fn next(&mut self) -> Option<Weak<LiveRelation>> {
        let entry = &self.relations[&self.head?];
        self.head = Some(entry.next);
        Some(entry.relation.clone())
    }
}

impl IdlePublication {
    pub(crate) fn new(trace: bool) -> Self {
        Self { trace, ..Self::default() }
    }

    fn add(&self, relations: &[Rc<LiveRelation>]) {
        if relations.is_empty() {
            return;
        }
        let mut schedule = self.schedule.borrow_mut();
        for relation in relations {
            schedule.insert(Rc::downgrade(relation));
        }
        schedule.quiet_since = Some(Instant::now());
        schedule.remaining = schedule.relations.len();
    }

    pub(crate) fn activity(&self) {
        let mut schedule = self.schedule.borrow_mut();
        if !schedule.relations.is_empty() {
            schedule.quiet_since = Some(Instant::now());
            schedule.remaining = schedule.relations.len();
        }
    }

    pub(crate) fn step(&self, ip: &Interp) -> bool {
        let (next, charged) = {
            let mut schedule = self.schedule.borrow_mut();
            // Leave a gap for a following writer request before doing conversion.
            const QUIET: Duration = Duration::from_micros(250);
            if schedule.remaining == 0 || schedule.quiet_since.is_some_and(|since| since.elapsed() < QUIET) {
                return false;
            }
            if let Some(lease) = &schedule.lease {
                let next = schedule.relations[&lease.owner].relation.clone();
                let charged = lease.charged;
                schedule.remaining = 0;
                (next, charged)
            } else {
                let Some(next) = schedule.next() else { return false };
                schedule.remaining -= 1;
                (next, 0)
            }
        };
        if let Some(live) = next.upgrade() {
            let budget = IDLE_BYTES - PAGE_HEADROOM - charged;
            if let Some((start, end, bytes)) = live.relation.borrow_mut().idle_append(&live.conn, ip, budget) {
                let charged = charged + bytes;
                if self.trace {
                    eprintln!("sql-idle rows {start}..{end} charged {charged}");
                }
                // Keep the lease even when caught up. The prepared pages are
                // still speculative until ordinary refresh or owner removal.
                let mut schedule = self.schedule.borrow_mut();
                schedule.lease = Some(Lease { owner: Rc::as_ptr(&live), charged });
                schedule.remaining = 1;
            }
        }
        self.schedule.borrow().remaining != 0
    }
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| Error::typ("embedded NUL in a database name, path or SQL statement"))
}

struct Connection {
    db: db::duckdb_database,
    conn: db::duckdb_connection,
}

impl Connection {
    fn open(path: &str) -> Result<Self> {
        let path = cstring(path)?;
        let mut ret = Self { db: ptr::null_mut(), conn: ptr::null_mut() };
        let mut message = ptr::null_mut();
        unsafe {
            if db::duckdb_open_ext(path.as_ptr(), &mut ret.db, ptr::null_mut(), &mut message) != db::duckdb_state_DuckDBSuccess {
                let error = if message.is_null() {
                    Error::system("cannot open database")
                } else {
                    let text = CStr::from_ptr(message).to_string_lossy().into_owned();
                    db::duckdb_free(message.cast());
                    Error::system(text)
                };
                return Err(error);
            }
            if db::duckdb_connect(ret.db, &mut ret.conn) != db::duckdb_state_DuckDBSuccess {
                return Err(Error::system("cannot connect to database"));
            }
        }
        Ok(ret)
    }

    fn query(&self, sql: &str) -> Result<QueryResult> {
        let sql = cstring(sql)?;
        unsafe {
            let mut result = QueryResult(core::mem::zeroed());
            if db::duckdb_query(self.conn, sql.as_ptr(), &mut result.0) != db::duckdb_state_DuckDBSuccess {
                let msg = db::duckdb_result_error(&mut result.0);
                return Err(if msg.is_null() {
                    Error::system("DuckDB query failed")
                } else {
                    Error::system(CStr::from_ptr(msg).to_string_lossy().into_owned())
                });
            }
            Ok(result)
        }
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        unsafe {
            db::duckdb_disconnect(&mut self.conn);
            db::duckdb_close(&mut self.db);
        }
    }
}

struct QueryResult(db::duckdb_result);

impl Drop for QueryResult {
    fn drop(&mut self) {
        unsafe { db::duckdb_destroy_result(&mut self.0) }
    }
}

fn integer(v: i128) -> Result<Value> {
    if let Ok(i) = Int::try_from(v) {
        return Ok(Value::Int(i));
    }
    #[cfg(feature = "bigint")]
    {
        Ok(klong_core::big::norm(v.into()))
    }
    #[cfg(not(feature = "bigint"))]
    {
        Err(Error::range("SQL integer exceeds the build width"))
    }
}

fn unsigned_integer(v: u128) -> Result<Value> {
    if let Ok(v) = i128::try_from(v) {
        return integer(v);
    }
    #[cfg(feature = "bigint")]
    {
        Ok(klong_core::big::norm(v.into()))
    }
    #[cfg(not(feature = "bigint"))]
    {
        Err(Error::range("SQL integer exceeds the build width"))
    }
}

// A mixed integer/float row must retain both SQL types, even above 2^53.
fn pack(values: Vec<Value>) -> Result<Value> {
    if values.iter().any(|v| matches!(v, Value::Int(_))) && values.iter().any(|v| matches!(v, Value::Flt(_))) {
        return Value::raw_list(values);
    }
    Value::from_items(values)
}

impl QueryResult {
    fn values(mut self) -> Result<Value> {
        let cols = unsafe { db::duckdb_column_count(&mut self.0) };
        let mut types = klong_core::value::try_vec(cols as usize)?;
        for i in 0..cols {
            types.push(ResultType::from_result(&mut self.0, i));
        }
        // The physical result width must agree with the selected Klong build.
        if let [kind @ ResultType::Scalar(t)] = types.as_slice() {
            let int_type = if size_of::<Int>() == 8 {
                db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT
            } else {
                db::DUCKDB_TYPE_DUCKDB_TYPE_INTEGER
            };
            let float_type = if size_of::<Flt>() == 8 {
                db::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE
            } else {
                db::DUCKDB_TYPE_DUCKDB_TYPE_FLOAT
            };
            if *t == int_type {
                return unsafe { self.numeric_column(*kind, Value::Int, Value::ints) };
            }
            if *t == float_type {
                return unsafe { self.numeric_column(*kind, Value::Flt, Value::flts) };
            }
        }
        self.rows(&types, Vec::new())
    }

    /// T matches the SQL vector's physical type. Copy all-valid chunks straight
    /// into the Klong array. A NULL makes the whole result a mixed Klong list.
    unsafe fn numeric_column<T: Copy>(
        mut self, kind: ResultType, scalar: fn(T) -> Value, array: fn(Vec<T>) -> Value,
    ) -> Result<Value> {
        let mut packed = Vec::new();
        while let Some(chunk) = self.next_chunk()? {
            let len = chunk.len();
            if len == 0 {
                continue;
            }
            let col = chunk.column(0, kind)?;
            if !col.data.is_null() && col.all_valid(len) {
                packed.try_reserve(len).map_err(|_| Error::memory())?;
                packed.extend_from_slice(core::slice::from_raw_parts(col.data.cast::<T>(), len));
            } else {
                let capacity = packed.len().checked_add(len).ok_or_else(Error::memory)?;
                let mut rows = klong_core::value::try_vec(capacity)?;
                rows.extend(packed.into_iter().map(scalar));
                for row in 0..len {
                    rows.push(col.cell(row)?);
                }
                drop(chunk);
                return self.rows(&[kind], rows);
            }
        }
        Ok(match packed.len() {
            0 => Value::empty(),
            1 => scalar(packed[0]),
            _ => array(packed),
        })
    }

    fn rows(mut self, types: &[ResultType], mut rows: Vec<Value>) -> Result<Value> {
        while let Some(chunk) = self.next_chunk()? {
            let len = chunk.len();
            let columns = chunk.columns(types)?;
            rows.try_reserve(len).map_err(|_| Error::memory())?;
            // A single SQL column is already the sequence of result cells.
            if let [column] = columns.as_slice() {
                for r in 0..len {
                    rows.push(unsafe { column.cell(r)? });
                }
                continue;
            }
            for r in 0..len {
                let mut row = klong_core::value::try_vec(columns.len())?;
                for col in &columns {
                    // All vectors are owned by this chunk and have len rows.
                    row.push(unsafe { col.cell(r)? });
                }
                rows.push(pack(row)?);
            }
        }
        if rows.len() == 1 {
            Ok(rows.remove(0))
        } else {
            pack(rows)
        }
    }

    fn next_chunk(&mut self) -> Result<Option<Chunk>> {
        unsafe {
            let chunk = db::duckdb_fetch_chunk(self.0);
            if !chunk.is_null() {
                return Ok(Some(Chunk(chunk)));
            }
            let error = db::duckdb_result_error(&mut self.0);
            if error.is_null() {
                Ok(None)
            } else {
                Err(Error::system(CStr::from_ptr(error).to_string_lossy().into_owned()))
            }
        }
    }
}

#[derive(Clone, Copy)]
enum ResultType {
    Scalar(db::duckdb_type),
    Decimal { physical: db::duckdb_type, scale: u8 },
}

impl ResultType {
    fn from_result(res: &mut db::duckdb_result, col: u64) -> Self {
        unsafe {
            let kind = db::duckdb_column_type(res, col);
            if kind != db::DUCKDB_TYPE_DUCKDB_TYPE_DECIMAL {
                return Self::Scalar(kind);
            }
            let mut logical = db::duckdb_column_logical_type(res, col);
            let physical = db::duckdb_decimal_internal_type(logical);
            let scale = db::duckdb_decimal_scale(logical);
            db::duckdb_destroy_logical_type(&mut logical);
            Self::Decimal { physical, scale }
        }
    }
}

struct Chunk(db::duckdb_data_chunk);

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe { db::duckdb_destroy_data_chunk(&mut self.0) }
    }
}

impl Chunk {
    fn len(&self) -> usize {
        unsafe { db::duckdb_data_chunk_get_size(self.0) as usize }
    }

    fn column(&self, i: usize, kind: ResultType) -> Result<Cells<'_>> {
        unsafe {
            let vector = db::duckdb_data_chunk_get_vector(self.0, i as u64);
            if vector.is_null() {
                return Err(Error::system("missing SQL result vector"));
            }
            Ok(Cells {
                data: db::duckdb_vector_get_data(vector),
                validity: db::duckdb_vector_get_validity(vector),
                kind,
                _chunk: self,
            })
        }
    }

    fn columns<'a>(&'a self, types: &[ResultType]) -> Result<Vec<Cells<'a>>> {
        let mut columns = klong_core::value::try_vec(types.len())?;
        for (i, &kind) in types.iter().enumerate() {
            columns.push(self.column(i, kind)?);
        }
        Ok(columns)
    }
}

struct Cells<'a> {
    data: *mut core::ffi::c_void,
    validity: *mut u64,
    kind: ResultType,
    _chunk: &'a Chunk,
}

fn hugeint(v: db::duckdb_hugeint) -> i128 {
    ((v.upper as i128) << 64) | v.lower as i128
}

impl Cells<'_> {
    /// `len` must be the chunk length. DuckDB stores one validity bit per row, least significant first.
    unsafe fn all_valid(&self, len: usize) -> bool {
        if self.validity.is_null() {
            return true;
        }
        let words = unsafe { core::slice::from_raw_parts(self.validity, len.div_ceil(64)) };
        let full = len / 64;
        let tail = len % 64;
        words[..full].iter().all(|&word| word == u64::MAX)
            && (tail == 0 || words[full].trailing_ones() as usize >= tail)
    }

    /// `row` is below the chunk length and `T` matches the DuckDB vector's physical type.
    unsafe fn read<T: Copy>(&self, row: usize) -> T {
        self.data.cast::<T>().add(row).read()
    }

    unsafe fn signed(&self, row: usize, physical: db::duckdb_type) -> Result<i128> {
        Ok(match physical {
            db::DUCKDB_TYPE_DUCKDB_TYPE_TINYINT => self.read::<i8>(row) as i128,
            db::DUCKDB_TYPE_DUCKDB_TYPE_SMALLINT => self.read::<i16>(row) as i128,
            db::DUCKDB_TYPE_DUCKDB_TYPE_INTEGER => self.read::<i32>(row) as i128,
            db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT => self.read::<i64>(row) as i128,
            db::DUCKDB_TYPE_DUCKDB_TYPE_HUGEINT => hugeint(self.read(row)),
            _ => return Err(Error::system("invalid SQL integer storage")),
        })
    }

    /// `row` is below the chunk length.
    unsafe fn cell(&self, row: usize) -> Result<Value> {
        if !self.validity.is_null() && !db::duckdb_validity_row_is_valid(self.validity, row as u64) {
            return Ok(Value::Undefined);
        }
        if self.data.is_null() {
            return Err(Error::system("missing SQL result data"));
        }
        let t = match self.kind {
            ResultType::Decimal { physical, scale } => {
                let mantissa = self.signed(row, physical)?;
                return if scale == 0 {
                    integer(mantissa)
                } else {
                    Ok(Value::Flt((mantissa as f64 / 10_f64.powi(i32::from(scale))) as Flt))
                };
            }
            ResultType::Scalar(t) => t,
        };
        match t {
            db::DUCKDB_TYPE_DUCKDB_TYPE_BOOLEAN => Ok(Value::bool(self.read::<u8>(row) != 0)),
            db::DUCKDB_TYPE_DUCKDB_TYPE_TINYINT | db::DUCKDB_TYPE_DUCKDB_TYPE_SMALLINT | db::DUCKDB_TYPE_DUCKDB_TYPE_INTEGER
            | db::DUCKDB_TYPE_DUCKDB_TYPE_BIGINT | db::DUCKDB_TYPE_DUCKDB_TYPE_HUGEINT => integer(self.signed(row, t)?),
            db::DUCKDB_TYPE_DUCKDB_TYPE_UTINYINT => integer(self.read::<u8>(row) as i128),
            db::DUCKDB_TYPE_DUCKDB_TYPE_USMALLINT => integer(self.read::<u16>(row) as i128),
            db::DUCKDB_TYPE_DUCKDB_TYPE_UINTEGER => integer(self.read::<u32>(row) as i128),
            db::DUCKDB_TYPE_DUCKDB_TYPE_UBIGINT => integer(self.read::<u64>(row) as i128),
            db::DUCKDB_TYPE_DUCKDB_TYPE_UHUGEINT => {
                let v = self.read::<db::duckdb_uhugeint>(row);
                unsigned_integer(((v.upper as u128) << 64) | v.lower as u128)
            }
            db::DUCKDB_TYPE_DUCKDB_TYPE_FLOAT => Ok(Value::Flt(self.read::<f32>(row) as Flt)),
            db::DUCKDB_TYPE_DUCKDB_TYPE_DOUBLE => Ok(Value::Flt(self.read::<f64>(row) as Flt)),
            db::DUCKDB_TYPE_DUCKDB_TYPE_VARCHAR => self.text(row),
            db::DUCKDB_TYPE_DUCKDB_TYPE_DATE | db::DUCKDB_TYPE_DUCKDB_TYPE_TIME | db::DUCKDB_TYPE_DUCKDB_TYPE_TIME_NS
            | db::DUCKDB_TYPE_DUCKDB_TYPE_TIME_TZ | db::DUCKDB_TYPE_DUCKDB_TYPE_INTERVAL | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP
            | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_S | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_MS | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_NS
            | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_TZ | db::DUCKDB_TYPE_DUCKDB_TYPE_TIMESTAMP_TZ_NS => {
                Err(Error::typ("SQL temporal results require an explicit conversion")
                    .with_hint("use CAST(value AS VARCHAR), or epoch_ns(value) for timestamps"))
            }
            _ => Err(Error::typ(format!("unsupported SQL result type {t}"))),
        }
    }

    unsafe fn text(&self, row: usize) -> Result<Value> {
        // The pointer may refer to inline storage in `value` or text owned by the chunk.
        let mut value = self.read::<db::duckdb_string_t>(row);
        let length = db::duckdb_string_t_length(value) as usize;
        if length == 0 {
            return Value::text("");
        }
        let bytes = db::duckdb_string_t_data(&mut value).cast::<u8>();
        if bytes.is_null() {
            return Err(Error::system("DuckDB returned an invalid string"));
        }
        let bytes = core::slice::from_raw_parts(bytes, length);
        Value::text(core::str::from_utf8(bytes).map_err(|_| Error::typ("invalid UTF-8 in SQL text"))?)
    }
}

#[cfg(test)]
mod idle_tests {
    use super::*;
    use klong_core::{table::Table, value::Dict, Config, NoHooks};

    fn scheduler_fixture() -> (Rc<IdlePublication>, Interp, Value, Value) {
        let idle = Rc::new(IdlePublication::new(false));
        let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
        register(&mut ip, Some(idle.clone()));
        let table = Table::new(vec![(Value::text("n").unwrap(), Value::ints(vec![1]))])
            .unwrap().into_value();
        let mut tables = Dict::default();
        tables.set(Value::text("t").unwrap(), table).unwrap();
        let arg = Value::dict(tables);
        let db = ip.global(".db").unwrap();
        (idle, ip, db, arg)
    }

    #[test]
    fn repeated_database_drop_does_not_accumulate_idle_registrations() {
        let (idle, mut ip, db, arg) = scheduler_fixture();
        let keep = ip.apply(&db, vec![arg.clone()], Some(1)).unwrap();
        for _ in 0..64 {
            let temporary = ip.apply(&db, vec![arg.clone()], Some(1)).unwrap();
            drop(temporary);
            assert_eq!(idle.schedule.borrow().relations.len(), 1);
        }
        drop(keep);
        let schedule = idle.schedule.borrow();
        assert!(schedule.relations.is_empty());
        assert_eq!(schedule.relations.capacity(), 0);
        assert_eq!(schedule.remaining, 0);
    }

    #[test]
    fn rotation_stays_fair_across_database_removal_and_enrollment() {
        let (idle, mut ip, db, arg) = scheduler_fixture();
        let first = ip.apply(&db, vec![arg.clone()], Some(1)).unwrap();
        let second = ip.apply(&db, vec![arg.clone()], Some(1)).unwrap();
        let third = ip.apply(&db, vec![arg.clone()], Some(1)).unwrap();
        let (first_id, second_id, third_id) = {
            let mut schedule = idle.schedule.borrow_mut();
            let first_id = schedule.next().unwrap().as_ptr();
            let second_id = schedule.next().unwrap().as_ptr();
            let third_id = schedule.next().unwrap().as_ptr();
            assert_ne!(first_id, second_id);
            assert_ne!(second_id, third_id);
            assert_ne!(third_id, first_id);
            assert_eq!(schedule.next().unwrap().as_ptr(), first_id);
            (first_id, second_id, third_id)
        };
        // Drop the next relation and add a replacement before polling again.
        drop(second);
        assert!(!idle.schedule.borrow().relations.contains_key(&second_id));
        let replacement = ip.apply(&db, vec![arg], Some(1)).unwrap();
        {
            let mut schedule = idle.schedule.borrow_mut();
            assert_eq!(schedule.relations.len(), 3);
            assert_eq!(schedule.next().unwrap().as_ptr(), third_id);
            assert_eq!(schedule.next().unwrap().as_ptr(), first_id);
            let replacement_id = schedule.next().unwrap().as_ptr();
            assert_ne!(replacement_id, third_id);
            assert_ne!(replacement_id, first_id);
            assert_eq!(schedule.next().unwrap().as_ptr(), third_id);
        }
        drop((first, third, replacement));
        assert_eq!(idle.schedule.borrow().relations.len(), 0);
    }
}
