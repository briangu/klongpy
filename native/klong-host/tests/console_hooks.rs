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

#[test]
fn read_on_a_live_console_asks_for_one_line_at_a_time_and_gives_the_rest_back() {
    let mut p = HostPlatform::new(Console::Buffer(Vec::new()));
    let asked = Rc::new(RefCell::new(0usize));
    let a = asked.clone();
    let mut lines = vec!["[1 2] [3 4]\n".to_string(), "[5\n".to_string(), "6]\n".to_string(), "never\n".to_string()].into_iter();
    p.source = Some(Box::new(move |buf| {
        *a.borrow_mut() += 1;
        match lines.next() {
            Some(l) => {
                buf[..l.len()].copy_from_slice(l.as_bytes());
                l.len()
            }
            None => 0,
        }
    }));
    let mut rt = Runtime::new(p, Config::default(), None);
    let v = rt.run(".r()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "[1 2]");
    // the first line was asked for once; the second object is already in hand
    assert_eq!(*asked.borrow(), 1);
    let v = rt.run(".r()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "[3 4]");
    assert_eq!(*asked.borrow(), 1);
    // an object that spans lines asks for the lines it needs, and no more
    let v = rt.run(".r()", "<t>").unwrap().unwrap();
    assert_eq!(rt.interp.format(&v), "[5 6]");
    assert_eq!(*asked.borrow(), 3);
    // what was read from the host and not used can be handed back to it
    assert_eq!(rt.take_console_unread(), b"\n");
    assert_eq!(rt.take_console_unread(), b"");
}
