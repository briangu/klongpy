//! A function goes out as its source text: `{x+y}` for a classic function and `fn(a;b)::{a+b}` for one with named
//! parameters (`fn` is a placeholder the receiver does not define); a projection and an operator go out as their
//! display form.
use klong_core::{Config, Interp, NoHooks};
use klong_ipc::codec::{self, ext_ref, KdbLambda};

fn text_of(ip: &mut Interp, src: &str) -> String {
    let v = ip.run(src, "<t>").unwrap().unwrap();
    let mut out = Vec::new();
    codec::encode(&ip.syms, &v, &mut out).unwrap();
    let back = codec::decode(&out, true, &mut ip.syms).unwrap_or_else(|_| panic!("{src}: does not decode"));
    let l = ext_ref::<KdbLambda>(&back).unwrap_or_else(|| panic!("{src}: not a lambda"));
    assert!(l.is_klong());
    l.text_str().unwrap().to_string()
}

#[test]
fn the_text_a_function_goes_out_as() {
    let mut ip = Interp::new(Config::default(), Box::new(NoHooks));
    let cases = [
        ("{x+1}", "{x+1}"),
        ("{x*y}", "{x*y}"),
        ("f0()::{7}", "fn()::{7}"),
        ("f1(a)::{a+1}", "fn(a)::{a+1}"),
        ("f2(a;b)::{a-b}", "fn(a;b)::{a-b}"),
        ("f6(a;b;c;d;e;g)::{a+b+c+d+e+g}", "fn(a;b;c;d;e;g)::{a+b+c+d+e+g}"),
        ("lc(a)::{[t];t::a*2;t+1}", "fn(a)::{[t];t::a*2;t+1}"),
        ("hn(p)::{[a b]::p;a-b}", "fn(p)::{[a b]::p;a-b}"),
        ("rn(a)::{:[a<1;0;a+.f(a-1)]}", "fn(a)::{:[a<1;0;a+.f(a-1)]}"),
        ("g::{x+y};g(1;)", ":monad"),
        ("{x+y}(1;)", ":monad"),
        ("f2(1;)", ":monad"),
    ];
    for (src, want) in cases {
        assert_eq!(text_of(&mut ip, src), want, "{src}");
    }
}
