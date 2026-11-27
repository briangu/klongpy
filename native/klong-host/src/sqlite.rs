//! A local SQLite connection. A call executes one SQL statement and returns rows.
use crate::local_file::{create_file, io_error};
use klong_core::{
    value::{try_vec, Func},
    Error, Flt, Int, Interp, Value,
};
use rusqlite::{
    fallible_iterator::FallibleIterator, types::ValueRef, Batch, Connection, OpenFlags,
};
use std::{path::Path, rc::Rc, time::Duration};

pub(super) fn register(ip: &mut Interp) {
    ip.register(
        ".sqlite",
        1,
        Rc::new(|ip, args| {
            let [Value::Str(path)] = args else {
                return Err(Error::typ("expected SQLite database path"));
            };
            if path.as_str().is_empty() {
                return Err(Error::range("empty SQLite database path"));
            }
            if path.as_str() != ":memory:" {
                create_file(Path::new(path.as_str())).map_err(io_error)?;
            }
            let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE;
            let db = Connection::open_with_flags(path.as_str(), flags).map_err(io_error)?;
            db.busy_timeout(Duration::ZERO).map_err(io_error)?;
            let f = Rc::new(move |_: &mut Interp, args: &[Value]| {
                let [Value::Str(sql)] = args else {
                    return Err(Error::typ("expected SQL text"));
                };
                let mut statements = Batch::new(&db, sql.as_str());
                let Some(mut stmt) = statements.next().map_err(io_error)? else {
                    return Ok(Value::empty());
                };
                if statements.next().map_err(io_error)?.is_some() {
                    return Err(Error::range("expected one SQLite statement"));
                }
                let n = stmt.column_count();
                let mut cursor = stmt.query([]).map_err(io_error)?;
                let mut rows = Vec::new();
                while let Some(row) = cursor.next().map_err(io_error)? {
                    let mut values = try_vec(n)?;
                    for i in 0..n {
                        values.push(match row.get_ref(i).map_err(io_error)? {
                            ValueRef::Null => Value::Undefined,
                            ValueRef::Integer(v) => Value::Int(
                                Int::try_from(v)
                                    .map_err(|_| Error::range("integer exceeds runtime width"))?,
                            ),
                            ValueRef::Real(v) => Value::Flt(v as Flt),
                            ValueRef::Text(v) => {
                                Value::text(std::str::from_utf8(v).map_err(io_error)?)?
                            }
                            ValueRef::Blob(v) => {
                                Value::ints(v.iter().map(|&b| Int::from(b)).collect())
                            }
                        });
                    }
                    rows.push(Value::raw_list(values)?);
                }
                Value::raw_list(rows)
            });
            Ok(Value::Fn(Rc::new(Func::native(
                ip.syms.intern(".sqlite.query"),
                1,
                f,
            ))))
        }),
    );
}
