//! The embedder takes the console: output at each write, input when a program reads the console channel.
use std::cell::RefCell;
use std::rc::Rc;

use klong_core::Config;
use klong_host::{Console, HostPlatform};
use klong_rt::Runtime;

#[test]
fn output_goes_to_the_sink_at_each_write_and_input_comes_from_the_source() {
    let mut p = HostPlatform::new(Console::Buffer(Vec::new()));
    let out: Rc<RefCell<Vec<(String, bool)>>> = Rc::new(RefCell::new(Vec::new()));
    let o = out.clone();
    p.sink = Some(Box::new(move |b, err| o.borrow_mut().push((String::from_utf8_lossy(b).into_owned(), err))));
    let mut lines = vec!["12 13\n".to_string(), "second\n".to_string()].into_iter();
    p.source = Some(Box::new(move |buf| match lines.next() {
        Some(l) => {
            buf[..l.len()].copy_from_slice(l.as_bytes());
            l.len()
        }
        None => 0,
    }));
    let mut rt = Runtime::new(p, Config::default(), None);
    let v = rt.run(".p(\"a\");.d(\"b\");.rl()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "\"12 13\"");
    let v = rt.run(".rl()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "\"second\"");
    // the end of the input
    let v = rt.run(".rl()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "\"\"");
    let got = out.borrow().iter().map(|(s, _)| s.clone()).collect::<Vec<_>>().concat();
    assert_eq!(got, "a\nb");
}
