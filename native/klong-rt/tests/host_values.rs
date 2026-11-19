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
