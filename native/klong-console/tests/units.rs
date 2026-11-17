//! The parts of the console that need no board: CRC-32, Intel HEX records, the completeness
//! check, the line editor.
use klong_console::complete::{check, Verdict};
use klong_console::crc::{crc32, Crc32};
use klong_console::edit::{Editor, Event};
use klong_console::hex::{self, HexError, Record};

#[test]
fn crc32_is_zlibs() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
    let mut c = Crc32::new();
    c.update(b"1234");
    c.update(b"");
    c.update(b"56789");
    assert_eq!(c.finish(), 0xCBF4_3926);
}

fn parse(line: &str) -> Result<(u8, u16, Vec<u8>), HexError> {
    let mut buf = [0u8; hex::MAX_DATA];
    hex::parse(line.as_bytes(), &mut buf).map(|r| match r {
        Record::Data { addr, data } => (0, addr, data.to_vec()),
        Record::Eof => (1, 0, vec![]),
        Record::ExtSegment(s) => (2, s, vec![]),
        Record::ExtLinear(u) => (4, u, vec![]),
        Record::Start => (3, 0, vec![]),
    })
}

#[test]
fn hex_records_of_the_reference() {
    // the worked examples of docs/native/porting.md
    let data: Vec<u8> = (1..=16).collect();
    assert_eq!(parse(":100000000102030405060708090A0B0C0D0E0F1068"), Ok((0, 0, data)));
    assert_eq!(parse(":020000040001F9"), Ok((4, 1, vec![])));
    assert_eq!(parse(":0400000508000000EF"), Ok((3, 0, vec![])));
    assert_eq!(parse(":00000001FF"), Ok((1, 0, vec![])));
    assert_eq!(parse(":0300300002337A1E"), Ok((0, 0x30, vec![0x02, 0x33, 0x7A])));
    // lower case, a carriage return, blanks around
    assert_eq!(parse("  :0300300002337a1e\r"), Ok((0, 0x30, vec![0x02, 0x33, 0x7A])));
    // extended segment address and a start segment address
    assert_eq!(parse(":020000021200EA"), Ok((2, 0x1200, vec![])));
    assert_eq!(parse(":0400000300003800C1"), Ok((3, 0, vec![])));
    // an empty data record
    assert_eq!(parse(":00001000F0"), Ok((0, 0x10, vec![])));
}

#[test]
fn hex_every_error() {
    assert_eq!(parse(""), Err(HexError::NoColon));
    assert_eq!(parse("00000001FF"), Err(HexError::NoColon));
    assert_eq!(parse(":"), Err(HexError::Short));
    assert_eq!(parse(":00"), Err(HexError::Short));
    assert_eq!(parse(":0000000"), Err(HexError::Digit));
    assert_eq!(parse(":0000000G"), Err(HexError::Digit));
    assert_eq!(parse(":00000001F"), Err(HexError::Digit));
    assert_eq!(parse(":0000 001FF"), Err(HexError::Digit));
    // count says 3, the line carries 2 or 4
    assert_eq!(parse(":03003000023300"), Err(HexError::Length));
    assert_eq!(parse(":0300300002337A7A00"), Err(HexError::Length));
    // one bit of the data, the address, the count, the check sum itself
    assert_eq!(parse(":0300300002337B1E"), Err(HexError::Checksum));
    assert_eq!(parse(":0300310002337A1E"), Err(HexError::Checksum));
    assert_eq!(parse(":0300300002337A1F"), Err(HexError::Checksum));
    // types beyond 05, with a good check sum
    assert_eq!(parse(":00000006FA"), Err(HexError::Type(6)));
    assert_eq!(parse(":000000FF01"), Err(HexError::Type(0xFF)));
    // fixed-size records of the wrong size
    assert_eq!(parse(":01000001AA54"), Err(HexError::Field));
    assert_eq!(parse(":0100000400FB"), Err(HexError::Field));
    assert_eq!(parse(":0100000200FD"), Err(HexError::Field));
    assert_eq!(parse(":020000050000F9"), Err(HexError::Field));
    // more than 255 data bytes cannot be said, and a line that long is a length error
    let long = format!(":FF000000{}00", "00".repeat(300));
    assert_eq!(parse(&long), Err(HexError::Length));
}

#[test]
fn hex_write_then_parse() {
    let mut line = [0u8; hex::MAX_LINE];
    for len in [0usize, 1, 16, 32, 255] {
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + len) as u8).collect();
        let n = hex::write(0, 0xBEEF, &data, &mut line);
        assert_eq!(n, 11 + 2 * len);
        let text = std::str::from_utf8(&line[..n]).unwrap();
        assert_eq!(parse(text), Ok((0, 0xBEEF, data)));
    }
    let n = hex::write(1, 0, &[], &mut line);
    assert_eq!(&line[..n], b":00000001FF");
    let n = hex::write(4, 0, &[0, 1], &mut line);
    assert_eq!(&line[..n], b":020000040001F9");
    // a buffer that is too small is left alone
    assert_eq!(hex::write(0, 0, &[1, 2, 3], &mut line[..10]), 0);
}

const C: Verdict = Verdict::Complete;
const I: Verdict = Verdict::Incomplete;
const E: Verdict = Verdict::Error;

/// CASES of tests/test_repl_complete.py, in the same order. (native/tests/test_console.py
/// runs the Python table itself, and random text, through `klong-sim --check-complete`.)
const CASES: &[(&str, Verdict)] = &[
    ("", C), ("   ", C), ("1+2", C), ("f(1;2)", C), ("\n", C),
    ("(", I), ("[", I), ("{", I), (":[", I), (":{", I), (":(", I),
    ("(1+2", I), ("arr::[1", I), ("f::{", I), ("f::{x+1", I), (":[1;2", I),
    ("{x}(", I), ("d:{[1 2]", I), ("f:(1", I), ("x:[1;2;", I),
    ("()", C), ("[]", C), ("{}", C), (":[1;2;3]", C), (":{[1 2]}", C), ("f::{x+1}", C),
    ("[1 2 3]", C), ("(1+2)", C), ("{x}(1)", C), (":[a;b;c]", C),
    ("f::{[a];a::(1+[2", I), ("f::{[a];a::(1+[2])}", C), ("{(}", E), ("{[}", E),
    ("f({x}(", I), (":[1;:[2;3]", I), (":[1;:[2;3]]", C), ("[[1 2][3", I),
    ("(1]", E), ("[1)", E), ("{1)", E), (")", E), ("]", E), ("}", E),
    ("1)", E), ("(1))", E), (":[1;2)", E), (":{1]", E), ("(1]\n", E), ("(}", E),
    ("[ ( ]", E), ("[ (", I), ("[1 (2) 3]", C), ("[1 [2 3] 4", I),
    ("\"abc", I), ("\"abc\"", C), ("\"a[b\"", C), ("\"a(b\"", C), ("\"a\"\"b\"", C), ("\"a\"\"", I),
    ("\"a{\"", C), ("f(\"(", I), ("f(\"(\")", C), ("\"", I), ("\"\"", C), ("\"a\nb", I),
    ("\"a\nb\"", C),
    ("0c[", C), ("0c(", C), ("0c{", C), ("0c]", C), ("[0c[ 0c]]", C), ("0c\"", C), ("x::0c[", C),
    (":\"a { b\"", C), (":\"a ( b\" 1", C), ("1 :\"x", I), ("1+2 :\"unfinished", I), (":\"", I),
    ("1 :\"x[\" ", C), ("(1 :\"ok ) bracket\"", I), ("(1 :\"ok ) bracket\")", C),
    ("f::{x :\"{\"", I), ("f::{x :\"{\"}", C),
    (".comment(\"X\")", I), (".comment(\"X\"); 1", I), (".comment(\"X\")\nbody (\nX", C),
    (".comment(\"X\")\nbody\nX\n1+2", C), (".comment(\"END\")\n[ ( {\n", I),
    (".comment(\"END\")\n[ ( {\nEND", C), (".comment(\"END\")\n[ ( {\nEND\n(1", I),
    ("1+.comment(\"X\")abc", I), (".comment(\"X\") abc X", C),
    ("1+", C), ("x@", C), ("f::", C), ("1+2*", C), ("-", C), ("#", C), ("a,", C), ("1 2 3+", C),
    ("1;", C), ("a::", C), ("+/", C),
    ("f::{x}\n", C), ("f::{\n", I), ("f::{\nx\n", I), ("f::{\nx\n}\n", C),
    ("f(\"abc", I), ("{\"abc", I),
];

#[test]
fn complete_table() {
    for (text, want) in CASES {
        assert_eq!(check(text), *want, "{:?}", text);
    }
}

#[test]
fn complete_joined_lines() {
    let mut text = String::from("f::{");
    assert_eq!(check(&text), I);
    text.push_str("\n x+1");
    assert_eq!(check(&text), I);
    text.push_str("\n}");
    assert_eq!(check(&text), C);
}

#[test]
fn complete_never_fails_on_noise() {
    let alphabet = ["(", ")", "[", "]", "{", "}", ":", "\"", "\\", " ", ";", "\n", "+", "-", "0", "c", ".", "a", "b", "1", ":[", ":{", ":(", ".comment(\"X\")", "é", "X"];
    let mut x = 7u64;
    let mut next = move |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    for _ in 0..20000 {
        let mut text = String::new();
        for _ in 0..next(15) {
            text.push_str(alphabet[next(alphabet.len())]);
        }
        let _ = check(&text);
    }
}

// ----- the line editor ---------------------------------------------------------------------

fn typed(ed: &mut Editor, keys: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    for &k in keys {
        assert_eq!(ed.key(k, &mut out), Event::None);
    }
    out
}

#[test]
fn editor_insert_and_backspace() {
    let mut ed = Editor::new(64, 4, 256, false);
    assert_eq!(typed(&mut ed, b"1+2"), b"1+2");
    assert_eq!(typed(&mut ed, b"\x7f"), b"\x08 \x08");
    assert_eq!(ed.buf, b"1+");
    assert_eq!(typed(&mut ed, b"\x08\x08\x08\x08"), b"\x08 \x08\x08 \x08");
    assert_eq!(ed.buf, b"");
    // a tab is a space; a control character shows as ^X and is kept
    assert_eq!(typed(&mut ed, b"a\tb\x01"), b"a b^A");
    assert_eq!(ed.buf, b"a b\x01");
    assert_eq!(typed(&mut ed, b"\x7f"), b"\x08 \x08\x08 \x08");
}

#[test]
fn editor_cursor_keys() {
    let mut ed = Editor::new(64, 4, 256, false);
    typed(&mut ed, b"abcd");
    // left twice, insert X: the tail is redrawn and the cursor comes back
    assert_eq!(typed(&mut ed, b"\x1b[D\x1b[D"), b"\x1b[1D\x1b[1D");
    assert_eq!(typed(&mut ed, b"X"), b"Xcd\x1b[2D");
    assert_eq!(ed.buf, b"abXcd");
    // backspace in the middle
    assert_eq!(typed(&mut ed, b"\x7f"), b"\x1b[1Dcd\x1b[K\x1b[2D");
    assert_eq!(ed.buf, b"abcd");
    // home, delete, end, right at the end does nothing
    assert_eq!(typed(&mut ed, b"\x1b[H"), b"\x1b[2D");
    assert_eq!(typed(&mut ed, b"\x1b[3~"), b"bcd\x1b[K\x1b[3D");
    assert_eq!(typed(&mut ed, b"\x1b[F"), b"\x1b[3C");
    assert_eq!(typed(&mut ed, b"\x1b[C"), b"");
    assert_eq!(ed.buf, b"bcd");
    // the other spellings of home and end, and a modifier parameter
    assert_eq!(typed(&mut ed, b"\x1b[1~"), b"\x1b[3D");
    assert_eq!(typed(&mut ed, b"\x1bOF"), b"\x1b[3C");
    assert_eq!(typed(&mut ed, b"\x1bOH"), b"\x1b[3D");
    assert_eq!(typed(&mut ed, b"\x1b[4~"), b"\x1b[3C");
    assert_eq!(typed(&mut ed, b"\x1b[1;5D"), b"\x1b[1D");
    // Ctrl-U empties the line
    assert_eq!(typed(&mut ed, b"\x15"), b"\x1b[2D\x1b[K");
    assert_eq!(ed.buf, b"");
}

#[test]
fn editor_utf8_moves_by_character() {
    let mut ed = Editor::new(64, 4, 256, false);
    // the bytes of a character are echoed together
    let mut out = Vec::new();
    for &k in "é".as_bytes() {
        ed.key(k, &mut out);
    }
    assert_eq!(out, "é".as_bytes());
    typed(&mut ed, "世x".as_bytes());
    assert_eq!(typed(&mut ed, b"\x1b[D\x1b[D"), b"\x1b[1D\x1b[1D");
    assert_eq!(typed(&mut ed, b"\x7f"), "\x1b[1D世x\x1b[K\x1b[2D".as_bytes());
    assert_eq!(ed.buf, "世x".as_bytes());
    // a broken sequence is kept as it came
    let mut ed = Editor::new(64, 4, 256, false);
    typed(&mut ed, b"\xC3A");
    assert_eq!(ed.buf, b"\xC3A");
    typed(&mut ed, b"\xE4\xB8");
    let mut out = Vec::new();
    assert_eq!(ed.finish(&mut out), Event::None);
    assert_eq!(ed.buf, b"\xC3A\xE4\xB8");
}

#[test]
fn editor_history() {
    let mut ed = Editor::new(64, 3, 16, false);
    for line in ["one", "two", "two", "", "three", "four"] {
        typed(&mut ed, line.as_bytes());
        ed.remember();
        ed.clear();
    }
    // three lines at most, no empty line, no repeat
    assert_eq!(ed.history_len(), 3);
    typed(&mut ed, b"new");
    assert_eq!(typed(&mut ed, b"\x1b[A"), b"\x1b[3D\x1b[Kfour");
    typed(&mut ed, b"\x1b[A\x1b[A");
    assert_eq!(ed.buf, b"two");
    typed(&mut ed, b"\x1b[A");
    assert_eq!(ed.buf, b"two");
    typed(&mut ed, b"\x1b[B\x1b[B");
    assert_eq!(ed.buf, b"four");
    // down past the newest gives the line that was being typed
    typed(&mut ed, b"\x1b[B");
    assert_eq!(ed.buf, b"new");
    typed(&mut ed, b"\x1b[B");
    assert_eq!(ed.buf, b"new");
    // the byte limit: a line longer than it is not kept, and old lines go
    ed.clear();
    typed(&mut ed, b"0123456789abcdefg");
    ed.remember();
    assert_eq!(ed.history_len(), 3);
    ed.clear();
    typed(&mut ed, b"0123456789abc");
    ed.remember();
    assert_eq!(ed.history_len(), 1);
    assert!(ed.history_bytes() <= 16);
}

#[test]
fn editor_bounds() {
    let mut ed = Editor::new(8, 4, 64, false);
    let mut out = Vec::new();
    for _ in 0..8 {
        assert_eq!(ed.key(b'x', &mut out), Event::None);
    }
    assert_eq!(ed.key(b'x', &mut out), Event::Overflow);
    assert_eq!(ed.buf.len(), 8);
    // an escape sequence without an end is given up after 16 bytes and nothing is inserted
    let mut ed = Editor::new(64, 4, 64, false);
    typed(&mut ed, b"\x1b[");
    typed(&mut ed, &[b'1'; 16]);
    assert!(ed.in_escape());
    typed(&mut ed, b"1");
    assert!(!ed.in_escape());
    assert_eq!(ed.buf, b"");
    // a control character ends an unfinished sequence and counts
    typed(&mut ed, b"q\x1b[1\x7f\x1b\x08");
    assert_eq!(ed.buf, b"");
    // an unknown sequence is swallowed whole
    typed(&mut ed, b"\x1b[200~a\x1b[Zb\x1bxc");
    assert_eq!(ed.buf, b"abc");
}

#[test]
fn editor_dumb() {
    let mut ed = Editor::new(8, 4, 64, true);
    let mut out = Vec::new();
    for &k in b"ab\x7fc\x1b\x02\td" {
        ed.key(k, &mut out);
    }
    assert_eq!(ed.buf, b"ac d");
    assert!(out.is_empty());
    ed.remember();
    assert_eq!(ed.history_len(), 0);
}
