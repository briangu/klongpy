use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1};
use klong_core::{
    par_add_inplace, par_sub_inplace, par_mul_inplace, par_div_inplace,
    par_add_inplace_f32, par_sub_inplace_f32, par_mul_inplace_f32, par_div_inplace_f32,
};
use std::sync::RwLock;

#[derive(Clone, Copy)]
enum RustDtype {
    F32,
    F64,
}

static DTYPE: RwLock<RustDtype> = RwLock::new(RustDtype::F64);

#[pyclass]
pub struct PyArrayF64 {
    inner: klong_core::Array<f64>,
}

#[pymethods]
impl PyArrayF64 {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<&'py pyo3::types::PyDict> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("shape", self.inner.shape.clone())?;
        dict.set_item("typestr", "<f8")?;
        dict.set_item("data", (self.inner.data.as_ptr() as usize, false))?;
        dict.set_item("strides", self.inner.strides.clone())?;
        Ok(dict)
    }
}

#[pyfunction]
fn add(a: &PyAny, b: &PyAny, py: Python) -> PyResult<PyObject> {
    match *DTYPE.read().unwrap() {
        RustDtype::F64 => {
            let a = a.downcast::<PyReadonlyArray1<f64>>()?;
            let b = b.downcast::<PyReadonlyArray1<f64>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f64>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_add_inplace(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
        RustDtype::F32 => {
            let a = a.downcast::<PyReadonlyArray1<f32>>()?;
            let b = b.downcast::<PyReadonlyArray1<f32>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f32>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_add_inplace_f32(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
    }
}

#[pyfunction]
fn subtract(a: &PyAny, b: &PyAny, py: Python) -> PyResult<PyObject> {
    match *DTYPE.read().unwrap() {
        RustDtype::F64 => {
            let a = a.downcast::<PyReadonlyArray1<f64>>()?;
            let b = b.downcast::<PyReadonlyArray1<f64>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f64>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_sub_inplace(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
        RustDtype::F32 => {
            let a = a.downcast::<PyReadonlyArray1<f32>>()?;
            let b = b.downcast::<PyReadonlyArray1<f32>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f32>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_sub_inplace_f32(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
    }
}

#[pyfunction]
fn multiply(a: &PyAny, b: &PyAny, py: Python) -> PyResult<PyObject> {
    match *DTYPE.read().unwrap() {
        RustDtype::F64 => {
            let a = a.downcast::<PyReadonlyArray1<f64>>()?;
            let b = b.downcast::<PyReadonlyArray1<f64>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f64>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_mul_inplace(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
        RustDtype::F32 => {
            let a = a.downcast::<PyReadonlyArray1<f32>>()?;
            let b = b.downcast::<PyReadonlyArray1<f32>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f32>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_mul_inplace_f32(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
    }
}

#[pyfunction]
fn divide(a: &PyAny, b: &PyAny, py: Python) -> PyResult<PyObject> {
    match *DTYPE.read().unwrap() {
        RustDtype::F64 => {
            let a = a.downcast::<PyReadonlyArray1<f64>>()?;
            let b = b.downcast::<PyReadonlyArray1<f64>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f64>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_div_inplace(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
        RustDtype::F32 => {
            let a = a.downcast::<PyReadonlyArray1<f32>>()?;
            let b = b.downcast::<PyReadonlyArray1<f32>>()?;
            if a.len() != b.len() {
                return Err(PyValueError::new_err("length mismatch"));
            }
            let out = PyArray1::<f32>::zeros(py, a.len(), false);
            let mut slice = unsafe { out.as_slice_mut().unwrap() };
            slice.copy_from_slice(a.as_slice()?);
            par_div_inplace_f32(&mut slice, b.as_slice()?);
            Ok(out.into_py(py))
        }
    }
}

#[pyfunction(name = "map")]
fn map_array(a: &PyAny, func: &PyAny, py: Python) -> PyResult<PyObject> {
    match *DTYPE.read().unwrap() {
        RustDtype::F64 => {
            let a = a.downcast::<PyReadonlyArray1<f64>>()?;
            let in_slice = a.as_slice()?;
            let out = PyArray1::<f64>::zeros(py, a.len(), false);
            let mut out_slice = unsafe { out.as_slice_mut().unwrap() };
            for (o, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                let val: f64 = func.call1((x,))?.extract()?;
                *o = val;
            }
            Ok(out.into_py(py))
        }
        RustDtype::F32 => {
            let a = a.downcast::<PyReadonlyArray1<f32>>()?;
            let in_slice = a.as_slice()?;
            let out = PyArray1::<f32>::zeros(py, a.len(), false);
            let mut out_slice = unsafe { out.as_slice_mut().unwrap() };
            for (o, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
                let val: f32 = func.call1((x,))?.extract()?;
                *o = val;
            }
            Ok(out.into_py(py))
        }
    }
}

#[pyfunction]
fn set_dtype(dtype: &str) -> PyResult<()> {
    let mut dt = DTYPE.write().unwrap();
    *dt = match dtype {
        "f32" | "float32" => RustDtype::F32,
        "f64" | "float64" => RustDtype::F64,
        _ => return Err(PyValueError::new_err("unsupported dtype")),
    };
    Ok(())
}

#[pyfunction]
fn get_dtype() -> PyResult<String> {
    let dt = DTYPE.read().unwrap();
    Ok(match *dt {
        RustDtype::F32 => "f32",
        RustDtype::F64 => "f64",
    }
    .to_string())
}

#[pyfunction]
fn to_pandas(rb: &PyAny, py: Python) -> PyResult<PyObject> {
    let pa = py.import("pyarrow")?;
    let table = pa.call_method1("Table.from_batches", (vec![rb],))?;
    let df = table.call_method0("to_pandas")?;
    Ok(df.into())
}

#[pymodule]
fn klongpy_rs(py: Python, m: &PyModule) -> PyResult<()> {
    pyo3::prepare_freethreaded_python();
    m.add_class::<PyArrayF64>()?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(subtract, m)?)?;
    m.add_function(wrap_pyfunction!(multiply, m)?)?;
    m.add_function(wrap_pyfunction!(divide, m)?)?;
    m.add_function(wrap_pyfunction!(map_array, m)?)?;
    m.add_function(wrap_pyfunction!(set_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(get_dtype, m)?)?;
    m.add_function(wrap_pyfunction!(to_pandas, m)?)?;
    Ok(())
}
