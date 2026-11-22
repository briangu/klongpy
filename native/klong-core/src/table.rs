//! Ordered, mutable columnar tables shared by the runtime and host integrations.
use alloc::boxed::Box;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::any::Any;
use core::cell::{Ref, RefCell};

use crate::error::Result;
use crate::value::{key_eq, take_deferred, try_push, try_vec, Ext, ExtBox, List};
use crate::{Error, Flt, Int, Interp, Value};

pub struct Table(RefCell<TableData>);

pub struct TableData {
    columns: Vec<(Value, Value)>,
    key_columns: Vec<usize>,
    rows: usize,
    revision: u64,
    last_rewrite_revision: u64,
    symbol_lookup: bool,
    metadata: Option<Rc<dyn Any>>,
}

enum AppendColumn<'a> {
    Ints(&'a mut Vec<Int>),
    Flts(&'a mut Vec<Flt>),
    List(&'a mut List),
}

impl TableData {
    /// The names (strings) and arrays of values, in schema order.
    pub fn columns(&self) -> &[(Value, Value)] {
        &self.columns
    }

    /// Positions of key columns, in key order.
    pub fn key_columns(&self) -> &[usize] {
        &self.key_columns
    }

    pub fn row_count(&self) -> usize {
        self.rows
    }

    /// Changes whenever the data, schema or key changes.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Last revision that could have changed an earlier row or the schema.
    pub fn last_rewrite_revision(&self) -> u64 {
        self.last_rewrite_revision
    }

    /// Host-supplied metadata about the original columns, which may predate the current revision.
    pub fn metadata(&self) -> Option<&dyn Any> {
        self.metadata.as_deref()
    }

    fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|(key, _)| matches!(key, Value::Str(s) if s.as_str() == name))
    }

    fn column(&self, ip: &Interp, name: &Value) -> Value {
        let name = match name {
            Value::Str(s) => s.as_str(),
            // q names the columns of a table with symbols.
            Value::Sym(s) if self.symbol_lookup => ip.syms.name(*s),
            _ => return Value::Undefined,
        };
        self.column_index(name).map(|i| self.columns[i].1.clone()).unwrap_or(Value::Undefined)
    }

    // A table owns its arrays until an evaluated column escapes. Reserve every
    // column before writing, then append in place to retain capacity for the next batch.
    fn append_in_place(&mut self, rows: &[Value], count: usize) -> Result<bool> {
        // Reading rank-one rows and mixed lists is allocation-free. Higher-rank
        // rows need materialization before mutating any column.
        if !self.key_columns.is_empty()
            || !rows.iter().all(|row| match row {
                Value::Ints(row) => row.rank() == 1,
                Value::Flts(row) => row.rank() == 1,
                Value::List(_) => true,
                _ => false,
            })
        {
            return Ok(false);
        }
        let mut targets = try_vec(self.columns.len())?;
        for (i, (_, column)) in self.columns.iter_mut().enumerate() {
            let target = match column {
                Value::Ints(a) if a.rank() == 1 && rows.iter().all(|row| matches!(row.item(i), Value::Int(_))) => {
                    let Some(a) = Rc::get_mut(a) else { return Ok(false) };
                    a.data.try_reserve(rows.len()).map_err(|_| Error::memory())?;
                    AppendColumn::Ints(&mut a.data)
                }
                Value::Flts(a) if a.rank() == 1 && rows.iter().all(|row| matches!(row.item(i), Value::Flt(_))) => {
                    let Some(a) = Rc::get_mut(a) else { return Ok(false) };
                    a.data.try_reserve(rows.len()).map_err(|_| Error::memory())?;
                    AppendColumn::Flts(&mut a.data)
                }
                Value::List(list) => {
                    let Some(list) = Rc::get_mut(list) else { return Ok(false) };
                    list.reserve_append(rows.iter().map(|row| row.item(i)))?;
                    AppendColumn::List(list)
                }
                _ => return Ok(false),
            };
            targets.push(target);
        }
        for (i, target) in targets.into_iter().enumerate() {
            match target {
                AppendColumn::Ints(a) => a.extend(rows.iter().filter_map(|row| row.item(i).as_int())),
                AppendColumn::Flts(a) => a.extend(rows.iter().filter_map(|row| row.item(i).as_flt())),
                AppendColumn::List(list) => list.append_reserved(rows.iter().map(|row| row.item(i))),
            }
        }
        self.rows = count;
        self.revision += 1;
        Ok(true)
    }
}

fn column_name(v: &Value) -> Result<&str> {
    match v {
        Value::Str(s) => Ok(s.as_str()),
        _ => Err(Error::typ("table column names must be strings")),
    }
}

fn column_len(v: &Value) -> Result<usize> {
    v.len().filter(|_| v.is_list()).ok_or_else(|| Error::typ("a table column must be a list of values"))
}

// Reading a row of a native numeric array can defer an allocation error. A mutation
// must observe that error before committing anything to the shared table.
fn checked_items(value: &Value) -> Result<Vec<Value>> {
    let items = value.items()?;
    match take_deferred() {
        Some(e) => Err(e),
        None => Ok(items),
    }
}

impl Table {
    pub fn new(columns: Vec<(Value, Value)>) -> Result<Self> {
        Self::from_columns(columns, 0, false)
    }

    /// Construct a table with the first `key_count` columns as keys, in input order.
    pub fn from_ipc(columns: Vec<(Value, Value)>, key_count: usize) -> Result<Self> {
        Self::from_columns(columns, key_count, true)
    }

    fn from_columns(columns: Vec<(Value, Value)>, key_count: usize, symbol_lookup: bool) -> Result<Self> {
        if key_count > columns.len() {
            return Err(Error::shape("a table cannot have more keys than columns"));
        }
        let mut rows = 0;
        for (i, (name, values)) in columns.iter().enumerate() {
            let name = column_name(name)?;
            let len = column_len(values)?;
            if columns[..i].iter().any(|(key, _)| matches!(key, Value::Str(s) if s.as_str() == name)) {
                return Err(Error::shape("table column names must be unique"));
            }
            if i == 0 {
                rows = len;
            } else if len != rows {
                return Err(Error::shape("table columns have different lengths"));
            }
        }
        let mut key_columns = try_vec(key_count)?;
        key_columns.extend(0..key_count);
        Ok(Self(RefCell::new(TableData { columns, key_columns, rows, revision: 0, last_rewrite_revision: 0, symbol_lookup, metadata: None })))
    }

    pub fn from_value(value: &Value) -> Option<&Self> {
        match value {
            Value::Ext(e) => e.0.as_any().downcast_ref(),
            _ => None,
        }
    }

    pub fn into_value(self) -> Value {
        Value::Ext(Rc::new(ExtBox(Box::new(self))))
    }

    pub fn data(&self) -> Ref<'_, TableData> {
        self.0.borrow()
    }

    pub fn set_metadata(&self, metadata: Rc<dyn Any>) {
        self.0.borrow_mut().metadata = Some(metadata);
    }

    pub fn set_index(&self, names: &Value) -> Result<Value> {
        if !names.is_list() || names.is_empty() {
            return Err(Error::typ(".index: expected a nonempty list of column names"));
        }
        let mut data = self.0.borrow_mut();
        if !data.key_columns.is_empty() {
            return Err(Error::typ(".index: the table already has an index"));
        }
        let names = checked_items(names)?;
        let mut keys = try_vec(names.len())?;
        for name in &names {
            let index = data.column_index(column_name(name)?).ok_or_else(|| Error::range(".index: unknown table column"))?;
            if keys.contains(&index) {
                return Err(Error::shape(".index: a key column may only be named once"));
            }
            keys.push(index);
        }
        let result = Value::raw_list(names)?;
        data.key_columns = keys;
        data.revision += 1;
        Ok(result)
    }

    pub fn reset_index(&self) -> bool {
        let mut data = self.0.borrow_mut();
        if data.key_columns.is_empty() {
            return false;
        }
        data.key_columns.clear();
        data.revision += 1;
        true
    }

    pub fn join_column(&self, pair: &Value) -> Result<()> {
        if !pair.is_list() || pair.len() != Some(2) {
            return Err(Error::typ(",: a table can be joined with a [name values] pair"));
        }
        let name = pair.item(0);
        let values = pair.item(1).unhost("table column")?;
        if let Some(e) = take_deferred() {
            return Err(e);
        }
        self.set_column(name, values)
    }

    pub fn set_column(&self, name: Value, values: Value) -> Result<()> {
        let column = column_name(&name)?;
        let len = column_len(&values)?;
        let mut data = self.0.borrow_mut();
        if !data.columns.is_empty() && len != data.rows {
            return Err(Error::shape("table columns have different lengths"));
        }
        if let Some(i) = data.column_index(column) {
            data.columns[i].1 = values;
        } else {
            try_push(&mut data.columns, (name, values))?;
        }
        data.rows = len;
        data.revision += 1;
        data.last_rewrite_revision = data.revision;
        Ok(())
    }

    pub fn insert(&self, input: &Value) -> Result<()> {
        if !input.is_list() {
            return Err(Error::typ(".insert: expected a row or a list of rows"));
        }
        let mut data = self.0.borrow_mut();
        let width = data.columns.len();
        if width == 0 {
            return Err(Error::shape(".insert: a table needs columns before inserting rows"));
        }
        let items = checked_items(input)?;
        let rows = if items.first().is_some_and(Value::is_list) {
            for row in &items {
                if !row.is_list() || row.len() != Some(width) {
                    return Err(Error::shape(".insert: row width differs from the number of columns"));
                }
            }
            items.as_slice()
        } else {
            if items.len() != width {
                return Err(Error::shape(".insert: row width differs from the number of columns"));
            }
            core::slice::from_ref(input)
        };

        // Work in temporary columns. All shape checks and allocations must finish
        // before any alias can observe the replacement.
        let capacity = data.rows.checked_add(rows.len()).ok_or_else(Error::memory)?;
        if data.append_in_place(&rows, capacity)? {
            return Ok(());
        }
        let mut columns = try_vec(width)?;
        for (_, column) in &data.columns {
            let mut values = try_vec(capacity)?;
            values.extend(checked_items(column)?);
            columns.push(values);
        }
        for row in rows {
            let row = checked_items(row)?;
            let position = if data.key_columns.is_empty() {
                None
            } else {
                (0..columns[0].len()).find(|&i| data.key_columns.iter().all(|&j| key_eq(&columns[j][i], &row[j])))
            };
            for (column, cell) in columns.iter_mut().zip(row) {
                match position {
                    Some(i) => column[i] = cell,
                    None => column.push(cell),
                }
            }
        }
        let count = columns[0].len();
        let mut updates = try_vec(width)?;
        for column in columns {
            updates.push(Value::from_items(column)?);
        }
        let mut rewrote_rows = !data.key_columns.is_empty();
        for ((_, column), update) in data.columns.iter_mut().zip(updates) {
            rewrote_rows |= core::mem::discriminant(column) != core::mem::discriminant(&update);
            *column = update;
        }
        data.rows = count;
        data.revision += 1;
        if rewrote_rows {
            data.last_rewrite_revision = data.revision;
        }
        Ok(())
    }
}

impl Ext for Table {
    fn type_name(&self) -> &str {
        "table"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> Option<Result<Value>> {
        let data = self.data();
        Some((|| {
            let mut names = try_vec(data.columns.len())?;
            names.extend(data.columns.iter().map(|(name, _)| name.clone()));
            Value::from_items(names)
        })())
    }

    fn find(&self, ip: &mut Interp, name: &Value) -> Option<Result<Value>> {
        let data = self.data();
        Some(if name.is_list() {
            (|| {
                let names = checked_items(name)?;
                let mut columns = try_vec(names.len())?;
                columns.extend(names.iter().map(|name| data.column(ip, name)));
                Value::from_items(columns)
            })()
        } else {
            Ok(data.column(ip, name))
        })
    }
}
