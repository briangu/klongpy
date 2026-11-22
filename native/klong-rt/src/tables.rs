//! System functions for portable Klong tables.
use alloc::rc::Rc;
use alloc::vec::Vec;

use klong_core::error::Result;
use klong_core::table::Table;
use klong_core::value::{take_deferred, try_vec};
use klong_core::{Error, Interp, Value};

use crate::one;

fn table_arg<'a>(value: &'a Value, name: &str) -> Result<&'a Table> {
    Table::from_value(value).ok_or_else(|| Error::typ(klong_core::msg!("{}: expected a table", name)))
}

fn two<'a>(args: &'a [Value], name: &str) -> Result<(&'a Value, &'a Value)> {
    match args {
        [x, y, ..] => Ok((x, y)),
        _ => Err(Error::arity(klong_core::msg!("{} takes 2 arguments, got {}", name, args.len()))),
    }
}

fn columns(value: &Value) -> Result<Vec<(Value, Value)>> {
    if let Value::Dict(dict) = value {
        let dict = dict.borrow();
        let mut columns = try_vec(dict.len())?;
        for (name, column) in dict.keys().iter().zip(dict.vals()) {
            columns.push((name.clone(), column.unhost(".table")?));
        }
        return Ok(columns);
    }

    let value = value.unhost(".table")?;
    let count = value.len().filter(|_| value.is_list()).ok_or_else(|| Error::typ(".table: expected a dictionary or list of [name values] pairs"))?;
    let mut columns = try_vec(count)?;
    for i in 0..count {
        let pair = value.item(i);
        if let Some(e) = take_deferred() {
            return Err(e);
        }
        if !pair.is_list() || pair.len() != Some(2) {
            return Err(Error::shape(".table: expected [name values] pairs"));
        }
        columns.push((pair.item(0), pair.item(1).unhost(".table")?));
        if let Some(e) = take_deferred() {
            return Err(e);
        }
    }
    Ok(columns)
}

pub fn register(interp: &mut Interp) {
    interp.register(
        ".table",
        1,
        Rc::new(|ip: &mut Interp, a: &[Value]| {
            let source = one(a, ".table")?;
            if let Some(result) = source.host_table(ip) {
                return result;
            }
            Ok(Table::new(columns(source)?)?.into_value())
        }),
    );
    interp.register(
        ".insert",
        2,
        Rc::new(|_: &mut Interp, a: &[Value]| {
            let (table, rows) = two(a, ".insert")?;
            table_arg(table, ".insert")?.insert(&rows.unhost(".insert")?)?;
            Ok(table.clone())
        }),
    );
    interp.register(
        ".index",
        2,
        Rc::new(|_: &mut Interp, a: &[Value]| {
            let (table, names) = two(a, ".index")?;
            table_arg(table, ".index")?.set_index(&names.unhost(".index")?)
        }),
    );
    interp.register(".rindex", 1, Rc::new(|_: &mut Interp, a: &[Value]| Ok(Value::bool(table_arg(one(a, ".rindex")?, ".rindex")?.reset_index()))));
    interp.register(
        ".schema",
        1,
        Rc::new(|_: &mut Interp, a: &[Value]| {
            if let Value::Ext(e) = one(a, ".schema")? {
                if let Some(result) = e.0.schema() {
                    return result;
                }
            }
            Err(Error::typ(".schema: expected a table or database"))
        }),
    );
}
