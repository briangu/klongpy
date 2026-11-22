//! q table integration for hosts with a mutable table value.
use alloc::rc::Rc;
use alloc::vec::Vec;

use klong_core::error::Result;
use klong_core::table::{Table, TableData};
use klong_core::value::{take_deferred, try_push, try_vec};
use klong_core::{Error, Int, Value};

use super::{perr, text, too_deep, DResult, Decoder, Encoder, MAX_DEPTH};

struct WireTable {
    body: Vec<u8>,
    column_types: Vec<u8>,
    mutable_columns: Vec<(usize, Vec<u8>)>,
}

// The table keeps its schema order when a column is replaced and appends new columns at the end.
fn wire_info(data: &TableData) -> Option<&WireTable> {
    data.metadata()?.downcast_ref()
}

/// q type byte for a column in the original wire schema, if one was received.
pub fn table_column_type(data: &TableData, index: usize) -> Option<u8> {
    wire_info(data)?.column_types.get(index).copied()
}

impl Encoder<'_> {
    pub(super) fn table(&mut self, table: &Table, depth: usize) -> Result<()> {
        let data = table.data();
        let keys = data.key_columns();
        let part_depth = depth + usize::from(!keys.is_empty());
        if part_depth + 1 >= MAX_DEPTH {
            return Err(too_deep());
        }
        let wire = wire_info(&data);
        if let Some(wire) = wire.filter(|_| data.revision() == 0) {
            // A dictionary or nested table can change without an operation on the outer table.
            let mut unchanged = true;
            for (i, original) in &wire.mutable_columns {
                let mut body = Vec::new();
                Encoder {
                    syms: self.syms,
                    out: &mut body,
                }
                .obj(&data.columns()[*i].1, part_depth + 2)?;
                if body != *original {
                    unchanged = false;
                    break;
                }
            }
            if unchanged {
                return self.put(&wire.body);
            }
        }
        let columns = data.columns();
        if !keys.is_empty() {
            self.put(&[99])?;
            self.table_part(columns, keys.iter().copied(), keys.len(), wire, part_depth)?;
        }
        let values = (0..columns.len()).filter(|i| !keys.contains(i));
        self.table_part(
            columns,
            values,
            columns.len() - keys.len(),
            wire,
            part_depth,
        )
    }

    fn table_part(
        &mut self,
        columns: &[(Value, Value)],
        positions: impl Iterator<Item = usize> + Clone,
        count: usize,
        wire: Option<&WireTable>,
        depth: usize,
    ) -> Result<()> {
        self.put(&[98, 0, 99])?;
        self.head(11, count)?;
        for i in positions.clone() {
            let Value::Str(name) = &columns[i].0 else {
                return Err(Error::internal("table name is not a string"));
            };
            let name = name.as_str().as_bytes();
            if name.contains(&0) {
                return Err(Error::typ("cannot send a column name containing NUL"));
            }
            self.put(name)?;
            self.put(&[0])?;
        }
        self.head(0, count)?;
        for i in positions {
            let column = &columns[i].1;
            let qtype = wire.and_then(|w| w.column_types.get(i)).copied();
            if let Some(t) = qtype {
                if self.typed_column(column, t, depth + 2)? {
                    continue;
                }
            }
            match column {
                Value::Ints(v) if v.rank() <= 1 => self.ints(&v.data)?,
                Value::Flts(v) if v.rank() <= 1 => self.flts(&v.data)?,
                _ => self.obj(column, depth + 2)?,
            }
        }
        Ok(())
    }

    fn typed_column(&mut self, column: &Value, t: u8, depth: usize) -> Result<bool> {
        let Some(len) = column.len() else {
            return Ok(false);
        };
        if t == 0 {
            return self.general(&table_items(column)?, depth).map(|()| true);
        }
        let start = self.out.len();
        self.head(t, len)?;
        for i in 0..len {
            let cell = column.item(i);
            if let Some(err) = take_deferred() {
                return Err(err);
            }
            if !self.typed_cell(&cell, t)? {
                self.out.truncate(start);
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn typed_cell(&mut self, cell: &Value, t: u8) -> Result<bool> {
        let null = matches!(cell, Value::Undefined);
        if t == 11 {
            return match cell {
                Value::Sym(s) => self.sym(*s).map(|()| true),
                Value::Undefined => self.put(&[0]).map(|()| true),
                _ => Ok(false),
            };
        }
        if t == 10 {
            return match cell {
                Value::Char(c) if (*c as u32) <= u8::MAX as u32 => {
                    self.put(&[*c as u8]).map(|()| true)
                }
                Value::Undefined => self.put(b" ").map(|()| true),
                _ => Ok(false),
            };
        }
        if t == 2 {
            if null {
                return self.put(&[0; 16]).map(|()| true);
            }
            let Value::Ints(bytes) = cell else {
                return Ok(false);
            };
            if bytes.rank() != 1 || bytes.data.len() != 16 {
                return Ok(false);
            }
            let mut bytes_out = [0; 16];
            for (input, output) in bytes.data.iter().zip(&mut bytes_out) {
                let Ok(value) = u8::try_from(*input) else {
                    return Ok(false);
                };
                *output = value;
            }
            return self.put(&bytes_out).map(|()| true);
        }
        if matches!(t, 8 | 9 | 15) {
            let f = if null {
                f64::from_bits(0xfff8_0000_0000_0000)
            } else if let Value::Flt(f) = cell {
                *f as f64
            } else {
                return Ok(false);
            };
            if t == 8 {
                let real = f as f32;
                if real as f64 != f && !f.is_nan() {
                    return Ok(false);
                }
                return self.put(&real.to_le_bytes()).map(|()| true);
            }
            return self.put(&f.to_le_bytes()).map(|()| true);
        }
        let Some(i) =
            cell.as_int()
                .map(|v| v as i64)
                .or_else(|| if null { null_int(t) } else { None })
        else {
            return Ok(false);
        };
        match t {
            1 if (0..=1).contains(&i) => self.put(&[i as u8])?,
            4 => {
                let Ok(byte) = u8::try_from(i) else {
                    return Ok(false);
                };
                self.put(&[byte])?;
            }
            5 => {
                let Ok(short) = i16::try_from(i) else {
                    return Ok(false);
                };
                self.put(&short.to_le_bytes())?;
            }
            6 | 13 | 14 | 17 | 18 | 19 => {
                let Ok(int) = i32::try_from(i) else {
                    return Ok(false);
                };
                self.put(&int.to_le_bytes())?;
            }
            7 | 12 | 16 => self.put(&i.to_le_bytes())?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

fn null_int(t: u8) -> Option<i64> {
    match t {
        5 => Some(i16::MIN.into()),
        6 | 13 | 14 | 17 | 18 | 19 => Some(i32::MIN.into()),
        7 | 12 | 16 => Some(i64::MIN),
        _ => None,
    }
}

fn table_items(column: &Value) -> Result<Vec<Value>> {
    let items = column.items()?;
    if let Some(err) = take_deferred() {
        return Err(err);
    }
    Ok(items)
}

fn holds_mutable_cells(column: &Value) -> bool {
    match column {
        Value::Dict(_) => true,
        Value::List(items) => items.iter().any(holds_mutable_cells),
        _ => Table::from_value(column).is_some(),
    }
}

impl Decoder<'_> {
    pub(super) fn table_value(&mut self, depth: usize, keyed: bool) -> DResult<Value> {
        let start = self.pos - 1;
        let mut columns = Vec::new();
        let mut column_types = Vec::new();
        if keyed {
            self.byte()?; // first table's type byte was peeked by dict()
        }
        let part_depth = depth + usize::from(keyed);
        self.table_columns(&mut columns, &mut column_types, part_depth)?;
        let key_count = if keyed { columns.len() } else { 0 };
        if keyed {
            if self.byte()? != 98 {
                return perr("keyed table values are not a table");
            }
            self.table_columns(&mut columns, &mut column_types, part_depth)?;
        }
        let mut mutable_columns = Vec::new();
        for (i, (_, column)) in columns.iter().enumerate() {
            if holds_mutable_cells(column) {
                let mut normalized = Vec::new();
                Encoder {
                    syms: self.syms,
                    out: &mut normalized,
                }
                .obj(column, part_depth + 2)?;
                try_push(&mut mutable_columns, (i, normalized))?;
            }
        }
        let table = Table::from_ipc(columns, key_count)?;
        let end = self.pos;
        let mut body = Vec::new();
        if self.little {
            super::put(&mut body, &self.buf[start..end])?;
        } else {
            self.pos = start;
            self.copy(&mut body, depth)?;
            self.pos = end;
        }
        table.set_metadata(Rc::new(WireTable {
            body,
            column_types,
            mutable_columns,
        }));
        Ok(table.into_value())
    }

    fn table_columns(
        &mut self,
        columns: &mut Vec<(Value, Value)>,
        column_types: &mut Vec<u8>,
        depth: usize,
    ) -> DResult<()> {
        if depth + 1 >= MAX_DEPTH {
            return perr("nesting deeper than 64");
        }
        self.byte()?; // table attribute
        if !matches!(self.byte()?, 99 | 127) {
            return perr("table without its dictionary");
        }
        if self.byte()? != 11 {
            return perr("table column names are not a symbol vector");
        }
        self.byte()?; // names attribute
        let n = self.count(1)?;
        let mut names = try_vec(n)?;
        for _ in 0..n {
            names.push(Value::str(text(self.cstring()?)?));
        }
        if self.byte()? != 0 {
            return perr("table columns are not a general list");
        }
        self.byte()?; // columns attribute
        if self.count(2)? != n {
            return perr("table names and columns differ in number");
        }
        for name in names {
            let t = self.peek()?;
            if t > 19 {
                return perr("table column is not a list");
            }
            let column = self.table_column(t, depth + 2)?;
            if columns
                .first()
                .is_some_and(|(_, first)| first.len() != column.len())
            {
                return perr("table columns differ in length");
            }
            try_push(columns, (name, column))?;
            try_push(column_types, t)?;
        }
        Ok(())
    }

    fn table_column(&mut self, t: u8, depth: usize) -> DResult<Value> {
        if t == 10 {
            self.byte()?; // type
            self.byte()?; // attribute
            let n = self.count(1)?;
            let mut items = try_vec(n)?;
            items.extend(self.take(n)?.iter().map(|&c| Value::Char(char::from(c))));
            return Ok(Value::raw_list(items)?);
        }
        let value = self.obj(depth)?;
        let missing_int = null_int(t).and_then(|v| Int::try_from(v).ok());
        let is_null = |cell: &Value| match cell {
            Value::Int(i) => missing_int == Some(*i),
            Value::Flt(f) => matches!(t, 8 | 9 | 15) && f.is_nan(),
            Value::Sym(s) => t == 11 && self.syms.name(*s).is_empty(),
            _ => false,
        };
        let missing = match &value {
            Value::Ints(v) => missing_int.is_some_and(|i| v.data.contains(&i)),
            Value::Flts(v) => matches!(t, 8 | 9 | 15) && v.data.iter().any(|f| f.is_nan()),
            Value::List(v) => t == 11 && v.iter().any(is_null),
            _ => false,
        };
        if !missing {
            return Ok(value);
        }
        let mut items = table_items(&value)?;
        for cell in &mut items {
            if is_null(cell) {
                *cell = Value::Undefined;
            }
        }
        Ok(Value::raw_list(items)?)
    }
}
