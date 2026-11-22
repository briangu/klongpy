//! Convert Python table types at the native table boundary.
use std::rc::Rc;

use klong_core::table::Table;
use klong_core::value::{max_nest, Dict};
use klong_core::{Interp, Value};
use klong_ipc::codec::table_column_type;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple, PyTypeMethods};

use super::{from_py, graph, k2py, to_py, Klong, PyTraverseError, PyVisit, Roots, ValueRef, ValueReference};

struct PythonSchema(ValueRef);

pub(super) fn python_schema(metadata: Option<&dyn std::any::Any>) -> Option<&ValueRef> {
    metadata?.downcast_ref::<PythonSchema>().map(|schema| &schema.0)
}

pub(super) fn import_table(py: Python<'_>, ip: &mut Interp, source: &Bound<'_, PyAny>) -> PyResult<Option<klong_core::error::Result<Value>>> {
    let source = py.import("klongpy.table_bridge")?.getattr("import_table")?.call1((source,))?;
    if source.is_none() {
        return Ok(None);
    }
    let (columns, schema): (Vec<(Bound<'_, PyAny>, Bound<'_, PyList>)>, Py<PyAny>) = source.extract()?;
    let mut native = Vec::with_capacity(columns.len());
    for (name, cells) in columns {
        let name = from_py(py, ip, &name)?;
        let values = cells.iter().map(|cell| import_cell(py, ip, &cell, 0)).collect::<PyResult<Vec<_>>>()?;
        native.push((name, pack(values)?));
    }
    let schema = ValueReference::new(py, schema)?;
    Ok(Some(Table::new(native).map(|table| {
        table.set_metadata(Rc::new(PythonSchema(schema)));
        table.into_value()
    })))
}

fn pack(values: Vec<Value>) -> PyResult<Value> {
    // Keep a mixed column exact even when an integer lies outside the exact range of f64.
    if values.iter().any(|v| matches!(v, Value::Int(_))) && values.iter().any(|v| matches!(v, Value::Flt(_))) {
        Value::raw_list(values).map_err(k2py)
    } else {
        Value::from_items(values).map_err(k2py)
    }
}

fn import_cell(py: Python<'_>, ip: &mut Interp, cell: &Bound<'_, PyAny>, depth: u32) -> PyResult<Value> {
    check_depth(depth)?;
    if cell.is_none() {
        return Ok(Value::Undefined);
    }
    if let Ok(list) = cell.downcast::<PyList>() {
        return pack(list.iter().map(|v| import_cell(py, ip, &v, depth + 1)).collect::<PyResult<Vec<_>>>()?);
    }
    if let Ok(tuple) = cell.downcast::<PyTuple>() {
        return pack(tuple.iter().map(|v| import_cell(py, ip, &v, depth + 1)).collect::<PyResult<Vec<_>>>()?);
    }
    if let Ok(dict) = cell.downcast::<PyDict>() {
        let mut out = Dict::default();
        for (name, value) in dict.iter() {
            out.set(from_py(py, ip, &name)?, import_cell(py, ip, &value, depth + 1)?).map_err(k2py)?;
        }
        return Ok(Value::dict(out));
    }
    from_py(py, ip, cell)
}

fn check_depth(depth: u32) -> PyResult<()> {
    if depth >= max_nest() {
        return Err(PyValueError::new_err("table cell nested too deeply"));
    }
    Ok(())
}

pub(super) fn is_ipc_table(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let ty = value.get_type();
    Ok(matches!(ty.name()?.to_str()?, "KdbTable" | "KdbKeyedTable") && ty.module()?.to_str()? == "klongpy.ipc.codec")
}

pub(super) fn import_ipc_table(py: Python<'_>, ip: &mut Interp, source: &Bound<'_, PyAny>) -> PyResult<Value> {
    let bytes = py.import("klongpy.ipc")?.getattr("encode")?.call1((source,))?;
    klong_ipc::decode(bytes.downcast::<PyBytes>()?.as_bytes(), true, &mut ip.syms).map_err(|e| match e {
        klong_ipc::DecodeError::Protocol(message) => PyTypeError::new_err(message),
        klong_ipc::DecodeError::Value(e) => k2py(e),
    })
}

pub(super) fn export_handle<'py>(py: Python<'py>, value: &Value, owner: Py<Klong>) -> PyResult<Bound<'py, PyAny>> {
    let wrap = Table::from_value(value).is_some_and(|table| {
        let data = table.data();
        data.key_columns().is_empty() && table_column_type(&data, 0).is_some()
    });
    let engine = owner.borrow(py).graph_id;
    let native = graph::registered(Bound::new(py, NativeTable { roots: Roots::default(), value: value.clone(), owner })?, engine, |handle| &handle.roots)?;
    if wrap {
        return py.import("klongpy.table_bridge")?.getattr("_WireTableView")?.call1((native,));
    }
    Ok(native.into_any())
}

/// A live, round-trippable view of a native table held by the interpreter.
#[pyclass(weakref)]
pub(super) struct NativeTable {
    pub(super) roots: Roots,
    pub(super) value: Value,
    pub(super) owner: Py<Klong>,
}

// The same interpreter/GIL discipline as NativeVec and NativeDict.
unsafe impl Send for NativeTable {}
unsafe impl Sync for NativeTable {}

impl NativeTable {
    fn table(&self) -> PyResult<&Table> {
        Table::from_value(&self.value).ok_or_else(|| PyTypeError::new_err("expected a native table"))
    }

    fn export<'py>(&self, py: Python<'py>, function: &str, dependency: &str) -> PyResult<Bound<'py, PyAny>> {
        let owner = self.owner.borrow(py);
        let (columns, schema) = owner.with_ip(py, |ip| -> PyResult<_> {
            let data = self.table()?.data();
            let columns = PyList::empty(py);
            for (i, (name, column)) in data.columns().iter().enumerate() {
                let values = PyList::empty(py);
                for j in 0..data.row_count() {
                    values.append(self.export_cell(py, ip, &column.item(j), 0)?)?;
                }
                columns.append((to_py(py, ip, name, Some(&self.owner))?, values, table_column_type(&data, i)))?;
            }
            let schema = data.metadata().and_then(|md| md.downcast_ref::<PythonSchema>()).map(|schema| schema.0.bind(py).map(Bound::unbind)).transpose()?;
            Ok((columns, schema))
        })?;
        py.import("klongpy.table_bridge")?.getattr(function)?.call1((py.import(dependency)?, columns, schema))
    }

    fn export_cell<'py>(&self, py: Python<'py>, ip: &Interp, value: &Value, depth: u32) -> PyResult<Bound<'py, PyAny>> {
        check_depth(depth)?;
        match value {
            Value::Undefined => Ok(py.None().into_bound(py)),
            Value::Dict(dict) => {
                let out = PyDict::new(py);
                let dict = dict.borrow();
                for (name, value) in dict.keys().iter().zip(dict.vals()) {
                    out.set_item(to_py(py, ip, name, Some(&self.owner))?, self.export_cell(py, ip, value, depth + 1)?)?;
                }
                Ok(out.into_any())
            }
            value if value.is_list() => {
                let out = PyList::empty(py);
                for i in 0..value.len().unwrap_or(0) {
                    out.append(self.export_cell(py, ip, &value.item(i), depth + 1)?)?;
                }
                Ok(out.into_any())
            }
            _ => to_py(py, ip, value, Some(&self.owner)),
        }
    }
}

#[pymethods]
impl NativeTable {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.roots.visit(&visit)?;
        visit.call(&self.owner)
    }

    fn __clear__(&self) {
        self.roots.clear();
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.table()?.data().row_count())
    }

    fn __repr__(&self) -> &'static str {
        ":table"
    }

    #[getter]
    fn columns(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.owner
            .borrow(py)
            .with_ip(py, |ip| self.table()?.data().columns().iter().map(|(name, _)| Ok(to_py(py, ip, name, Some(&self.owner))?.unbind())).collect())
    }

    #[getter]
    fn idx_cols(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyAny>>>> {
        self.owner.borrow(py).with_ip(py, |ip| {
            let data = self.table()?.data();
            let keys = data.key_columns();
            if keys.is_empty() {
                return Ok(None);
            }
            keys.iter().map(|&j| Ok(to_py(py, ip, &data.columns()[j].0, Some(&self.owner))?.unbind())).collect::<PyResult<_>>().map(Some)
        })
    }

    fn _ipc_column_type(&self, name: &str) -> PyResult<Option<u8>> {
        let data = self.table()?.data();
        let position = data.columns().iter().position(|(key, _)| matches!(key, Value::Str(key) if key.as_str() == name));
        Ok(position.and_then(|i| table_column_type(&data, i)))
    }

    fn __getitem__<'py>(slf: &Bound<'_, Self>, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        let view = slf.borrow();
        let owner = view.owner.borrow(py);
        view.roots.register(slf.as_any(), owner.graph_id)?;
        owner.with_ip(py, |ip| {
            let data = view.table()?.data();
            let column = data.columns().iter().find_map(|(name, value)| match name {
                Value::Str(name) if name.as_str() == key => Some(value),
                _ => None,
            });
            to_py(py, ip, column.unwrap_or(&Value::Undefined), Some(&view.owner))
        })
    }

    fn __setitem__(slf: &Bound<'_, Self>, py: Python<'_>, key: &str, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let view = slf.borrow();
        let owner = view.owner.borrow(py);
        view.roots.register(slf.as_any(), owner.graph_id)?;
        owner.with_ip(py, |ip| {
            let name = Value::text(key).map_err(k2py)?;
            let values = from_py(py, ip, values)?.unhost("table column").map_err(k2py)?;
            view.table()?.set_column(name, values).map_err(k2py)
        })
    }

    fn get_dataframe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.export(py, "export_dataframe", "pandas")
    }

    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(&self, py: Python<'py>, requested_schema: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        self.export(py, "export_arrow", "pyarrow")?.call_method1("__arrow_c_stream__", (requested_schema,))
    }

    fn __klong_ipc_body__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.owner.borrow(py).with_ip(py, |ip| {
            let mut bytes = Vec::new();
            klong_ipc::encode(&ip.syms, &self.value, &mut bytes).map_err(k2py)?;
            Ok(PyBytes::new(py, &bytes))
        })
    }
}
