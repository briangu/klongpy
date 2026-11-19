//! System functions on host arrays (the Python extension's tensors) that sit inside a list or a dictionary.
use klong_core::value::{Ext, ExtBox, Value};
use klong_core::{Config, Kind};
use klong_rt::memfs::MemFs;
use klong_rt::{FileSystem, Platform, Runtime};
use std::rc::Rc;

#[derive(Default)]
struct Board {
    fs: MemFs,
}

impl Platform for Board {
    fn now_us(&self) -> u64 {
        0
    }
    fn wall_clock(&self) -> Option<f64> {
        None
    }
    fn idle(&mut self) {}
    fn console_read(&mut self, _: &mut [u8]) -> usize {
        0
    }
    fn console_write(&mut self, _: &[u8]) {}
    fn fs(&mut self) -> Option<&mut dyn FileSystem> {
        Some(&mut self.fs)
    }
}

/// An array of the host: three numbers it hands over as a native list.
struct Tensor;
impl Ext for Tensor {
    fn type_name(&self) -> &str {
        "tensor"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_shape(&self) -> Option<Vec<u32>> {
        Some(vec![3])
    }
    fn host_native(&self, _what: &str) -> Option<klong_core::error::Result<Value>> {
        Some(Ok(Value::ints(vec![1, 2, 3])))
    }
    fn host_items(&self) -> Option<klong_core::error::Result<Value>> {
        Some(Ok(Value::ints(vec![1, 2, 3])))
    }
}

fn run_with_tensor(src: &str) -> String {
    let mut rt = Runtime::new(Board::default(), Config::default(), None);
    rt.interp.define_global("t", Value::Ext(Rc::new(ExtBox(Box::new(Tensor))))).unwrap();
    match rt.run(src, "<t>") {
        Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).unwrap(),
        Ok(None) => String::new(),
        Err(e) => format!("!{:?}", e.kind),
    }
}

#[test]
fn a_host_array_is_the_list_of_arguments_of_try_and_the_argument_of_float() {
    // `,1` under the torch backend is a host array, not a native list
    assert_eq!(run_with_tensor(".try({x+y+z};t)"), "[1 6]");
    assert_eq!(run_with_tensor(".float(t)"), "[1.0 2.0 3.0]");
}

#[test]
fn a_host_array_matches_inside_a_list_by_what_it_holds() {
    assert_eq!(run_with_tensor("[;t;4]~[[1 2 3] 4]"), "1");
    assert_eq!(run_with_tensor("[;t;4]~[[1 2 4] 4]"), "0");
}

#[test]
fn a_qualified_name_on_a_host_array_is_a_type_error_as_on_a_native_list() {
    assert_eq!(run_with_tensor("t.a"), format!("!{:?}", Kind::Type));
    assert_eq!(run_with_tensor("t.a::1"), format!("!{:?}", Kind::Type));
}

#[test]
fn json_writes_a_host_array_inside_a_dictionary_or_a_list() {
    let mut rt = Runtime::new(Board::default(), Config::default(), None);
    rt.interp.define_global("t", Value::Ext(Rc::new(ExtBox(Box::new(Tensor))))).unwrap();
    let show = |rt: &mut Runtime<Board>, src: &str| match rt.run(src, "<t>") {
        Ok(Some(v)) => klong_core::fmt::format_value(&rt.interp.syms, &v, false).unwrap(),
        Ok(None) => String::new(),
        Err(e) => format!("!{:?}", e.kind),
    };
    assert_eq!(show(&mut rt, ".json(t)"), "\"[1,2,3]\"");
    assert_eq!(show(&mut rt, ".json(:{;\"count\";3;\"values\";t})"), "\"{\"\"count\"\":3,\"\"values\"\":[1,2,3]}\"");
    assert_eq!(show(&mut rt, ".json([;1;t])"), "\"[1,[1,2,3]]\"");
    // a function is still not JSON
    assert_eq!(show(&mut rt, ".json(:{;\"f\";{x}})"), format!("!{:?}", Kind::Type));
}
