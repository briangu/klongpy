//! The frames of running functions share one array of slots: a call takes the next ones and gives them back when it returns, however it
//! ends. These tests run programs that fail at every depth and check that what runs after them is not affected.
use klong_core::{Config, Interp, Kind, NoHooks};

fn interp() -> Interp {
    Interp::new(Config::default(), Box::new(NoHooks))
}

fn run(ip: &mut Interp, text: &str) -> Result<String, Kind> {
    match ip.run(text, "<test>") {
        Ok(Some(v)) => Ok(ip.format(&v)),
        Ok(None) => Ok(String::new()),
        Err(e) => Err(e.kind),
    }
}

fn big<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(256 << 20).spawn(f).expect("thread").join().expect("join")
}

const DEFS: &str = "sq(a)::{a*a}\nbad(a)::{a+undefined_name}\nrec::{rec(x+1)}\nrecc::{[a];a::x;1+recc(x+1)}\nsum::{:[x<1;0;x+sum(x-1)]}\ncl::{x+y}\n";

#[test]
fn errors_at_any_depth_leave_the_next_calls_right() {
    big(|| {
        let mut ip = interp();
        run(&mut ip, DEFS).unwrap();
        let probes = [
            "bad(3)+sq(2)",
            "sq(bad(1))",
            "cl(1;bad(2))",
            "{bad(x)}'!5",
            "{:[x>3;bad(x);x+y]}/!10",
            "{sq(x)+bad(x)}'!3",
            "rec(0)",
            "recc(0)",
        ];
        for round in 0..30 {
            for p in probes {
                assert!(ip.run(p, "<test>").is_err(), "{} round {}", p, round);
                assert_eq!(run(&mut ip, "sq(7)+cl(2;3)+sum(10)"), Ok("109".into()), "after {} round {}", p, round);
            }
        }
    });
}

#[test]
fn an_assignment_among_the_arguments_of_a_call_is_kept() {
    // `a::1` creates a local of the running function while its call's first arguments wait on the slots above its frame
    let mut ip = interp();
    run(&mut ip, "cl::{x+y}\nf::{[b];b+cl(b::5;x)}\ng::{c+cl(1;c::x)}\nh::{c+cl(c::3;c::x)}").unwrap();
    assert_eq!(run(&mut ip, "f(10)"), Ok("20".into()));
    assert_eq!(run(&mut ip, "g(10)"), Ok("21".into()));
    assert_eq!(run(&mut ip, "h(10)"), Ok("23".into()));
}

#[test]
fn scopes_are_as_they_were() {
    let mut ip = interp();
    // named parameters: own frame, then globals; a classic function sees its caller's locals
    run(&mut ip, "n::100\nlex(a)::{a+n}\nouter::{[n];n::1;lex(5)}\ndyn::{x+n}\nouter2::{[n];n::1;dyn(5)}").unwrap();
    assert_eq!(run(&mut ip, "outer(0)"), Ok("105".into()));
    assert_eq!(run(&mut ip, "outer2(0)"), Ok("6".into()));
    assert_eq!(run(&mut ip, "n"), Ok("100".into()));
}
