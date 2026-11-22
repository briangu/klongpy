//! Every entry of `tests/ipc_vectors/vectors.json` (shared with KlongPy, `gen_vectors.py`
//! describes it) in its mode, through the codec and through the session:
//!
//! * `roundtrip`: decoding gives the value and encoding the value gives exactly the bytes;
//! * `decode`: decoding gives the value (big-endian, compressed, tables, temporals, widths);
//! * `reject`: the bytes are refused as a protocol error.
//!
//! `widths`: `64` entries run on the default build, `32` entries on the `int32,float32`
//! build (`cargo test -p klong-ipc --features klong-core/int32,klong-core/float32`), `any`
//! on both. Run with `--nocapture` for the counts and the entries that do not apply.
mod common;

use common::*;
use klong_core::value::Dict;
use klong_core::{Flt, Int, Interp, Value, FLT_BITS, INT_BITS};
use klong_ipc::codec::{self, DecodeError};
use klong_ipc::session::{Config, Event, Kind, Session};

fn vectors() -> Vec<Json> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/ipc_vectors/vectors.json");
    let text = std::fs::read_to_string(path).expect("tests/ipc_vectors/vectors.json");
    match parse_json(&text) {
        Json::Arr(a) => a,
        _ => panic!("vectors.json is not a list"),
    }
}

fn number(s: &str, float: bool) -> Value {
    if float {
        Value::Flt(s.parse::<f64>().unwrap() as Flt)
    } else {
        Value::Int(s.parse::<Int>().expect("an integer of this build's width"))
    }
}

fn special(s: &str) -> Flt {
    match s {
        "nan" => Flt::NAN,
        "inf" => Flt::INFINITY,
        "-inf" => Flt::NEG_INFINITY,
        other => panic!("float {:?}", other),
    }
}

/// A numeric array of the `expect` grammar: nested lists of numbers (or "inf" strings).
fn numeric(items: &[Json], float: bool) -> Value {
    let vals: Vec<Value> = items
        .iter()
        .map(|x| match x {
            Json::Arr(inner) => numeric(inner, float),
            Json::Num(n) => number(n, float),
            Json::Str(s) => Value::Flt(special(s)),
            other => panic!("array item {:?}", other),
        })
        .collect();
    if vals.is_empty() {
        return if float { Value::flts(Vec::new()) } else { Value::empty() };
    }
    Value::from_items(vals).unwrap()
}

/// The tagged JSON value of an entry's `expect` field (grammar: tests/test_ipc_codec.py).
fn from_expect(ip: &mut Interp, e: &Json) -> Value {
    if let Some(Json::Num(n)) = e.get("int") {
        return number(n, false);
    }
    if let Some(f) = e.get("float") {
        return match f {
            Json::Str(s) => Value::Flt(special(s)),
            Json::Num(n) => number(n, true),
            other => panic!("float {:?}", other),
        };
    }
    if let Some(s) = e.get("str") {
        return Value::str(s.str().to_string());
    }
    if let Some(c) = e.get("char") {
        return Value::Char(c.str().chars().next().unwrap());
    }
    if let Some(s) = e.get("sym") {
        return Value::Sym(ip.syms.intern(s.str()));
    }
    if e.get("undefined").is_some() {
        return Value::Undefined;
    }
    if let Some(t) = e.get("error") {
        return codec::error_value(t.str().as_bytes()).unwrap();
    }
    if let Some(l) = e.get("lambda") {
        return codec::lambda_value(l.arr()[0].str().as_bytes(), l.arr()[1].str().as_bytes()).unwrap();
    }
    if let Some(o) = e.get("opaque") {
        // the decoder is the only maker of opaque values: build one from its bytes
        let raw = unhex(o.arr()[1].str());
        return codec::decode(&raw, true, &mut ip.syms).unwrap();
    }
    if let Some(pairs) = e.get("dict") {
        let mut d = Dict::default();
        for kv in pairs.arr() {
            let k = from_expect(ip, &kv.arr()[0]);
            let v = from_expect(ip, &kv.arr()[1]);
            d.set(k, v).unwrap();
        }
        return Value::dict(d);
    }
    if let Some(items) = e.get("array") {
        let dt = e.get("dtype").unwrap().str();
        if dt == "object" {
            let vals: Vec<Value> = items.arr().iter().map(|x| from_expect(ip, x)).collect();
            return Value::from_items(vals).unwrap();
        }
        return numeric(items.arr(), dt.starts_with("float"));
    }
    panic!("expect {:?}", e)
}

fn ready_client() -> (Session, Interp) {
    let mut ip = interp();
    let mut s = Session::client(Config::default());
    s.hello("", "").unwrap();
    s.feed(&[3]).unwrap();
    assert!(matches!(s.next_event(&mut ip.syms), Some(Event::Accepted { capability: 3 })));
    (s, ip)
}

/// Feed a whole message to a ready session; the one event it makes.
fn through_session(msg: &[u8], ip: &mut Interp) -> Event {
    let (mut s, _) = ready_client();
    if msg.len() > 1 && msg[1] == 2 {
        // a response needs a request to answer
        s.send_sync(&ip.syms, &Value::Undefined, true, None, false).unwrap();
    }
    // in pieces, to exercise the buffering
    for chunk in msg.chunks(7) {
        s.feed(chunk).unwrap();
    }
    let ev = s.next_event(&mut ip.syms).unwrap_or(Event::Rejected);
    if matches!(ev, Event::Message { .. }) {
        assert!(s.next_event(&mut ip.syms).is_none(), "one message, one event");
    }
    ev
}

fn header_consistent(msg: &[u8]) -> bool {
    msg.len() >= 8 && msg[0] <= 1 && msg[1] <= 2 && msg[2] <= 1 && msg[3] == 0 && {
        let a = [msg[4], msg[5], msg[6], msg[7]];
        (if msg[0] == 1 { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }) as usize == msg.len()
    }
}

#[test]
fn ingest_call_decodes_symbols_and_long_vectors_in_both_byte_orders() {
    let symbols: [(&[u8], &str); 8] = [
        (b"S00", "S00"),
        (b"", ""),
        (b"xyA", "xyA"),
        (b"xya", "xya"),
        ("é".as_bytes(), "é"),
        (b"\xff", "\u{fffd}"),
        (b"\xfe", "\u{fffd}"),
        (b"S15", "S15"),
    ];
    let count = 256_i32;
    let bounds = [Int::MIN as i64, -1, 0, Int::MAX as i64];
    for little in [true, false] {
        let i32_bytes = |value: i32| if little { value.to_le_bytes() } else { value.to_be_bytes() };
        let i64_bytes = |value: i64| if little { value.to_le_bytes() } else { value.to_be_bytes() };
        let mut wire = vec![0, 0];
        wire.extend(i32_bytes(4));
        wire.extend(b"\xf5ingest\0");
        let symbol_vector_start = wire.len();
        wire.extend([11, 0]);
        wire.extend(i32_bytes(count));
        for (raw, _) in symbols.iter().cycle().take(count as usize) {
            wire.extend_from_slice(raw);
            wire.push(0);
        }
        let symbol_vector_end = wire.len();
        for values in [bounds, [11, 12, 13, 14]] {
            wire.extend([7, 0]);
            wire.extend(i32_bytes(count));
            for &cell in values.iter().cycle().take(count as usize) {
                wire.extend(i64_bytes(cell));
            }
        }

        let mut ip = interp();
        let value = codec::decode(&wire, little, &mut ip.syms).unwrap();
        let Value::List(args) = value else { panic!("expected a call envelope") };
        assert!(matches!(args[0], Value::Sym(s) if ip.syms.name(s) == "ingest"));
        let Value::List(col) = &args[1] else { panic!("expected symbols") };
        assert_eq!(col.len(), count as usize);
        for (cell, (_, expected)) in col.iter().zip(symbols.iter().cycle()) {
            assert!(matches!(cell, Value::Sym(s) if ip.syms.name(*s) == *expected));
        }
        for (col, values) in args[2..].iter().zip([bounds, [11, 12, 13, 14]]) {
            let Value::Ints(col) = col else { panic!("expected longs") };
            assert_eq!(col.data.len(), count as usize);
            assert!(col.data.iter().copied().map(|v| v as i64).eq(values.into_iter().cycle().take(count as usize)));
        }
        let truncated = &wire[symbol_vector_start..symbol_vector_end - 1];
        assert!(matches!(codec::decode(truncated, little, &mut ip.syms), Err(DecodeError::Protocol(_))));
    }
}

fn matches_decoded(value: &Value, expected: &Value, _ip: &Interp) -> bool {
    #[cfg(feature = "tables")]
    if let Some(table) = klong_core::table::Table::from_value(value) {
        let data = table.data();
        let columns = data.columns();
        let key_count = data.key_columns().len();
        if let Value::List(parts) = expected {
            return parts.len() == 2
                && data.key_columns().iter().copied().eq(0..key_count)
                && matches_columns(&columns[..key_count], &parts[0], _ip)
                && matches_columns(&columns[key_count..], &parts[1], _ip);
        }
        return key_count == 0 && matches_columns(columns, expected, _ip);
    }
    same(value, expected)
}

#[cfg(feature = "tables")]
fn matches_columns(columns: &[(Value, Value)], expected: &Value, ip: &Interp) -> bool {
    let Value::Dict(expected) = expected else { return false };
    let expected = expected.borrow();
    columns.len() == expected.len()
        && columns.iter().zip(expected.keys().iter().zip(expected.vals())).all(
            |((name, values), (key, wanted))| {
                matches!((name, key), (Value::Str(s), Value::Sym(k)) if s.as_str() == ip.syms.name(*k))
                    && same(values, wanted)
            }
        )
}

#[test]
fn every_vector_in_its_mode() {
    let wide = INT_BITS == 64 && FLT_BITS == 64;
    let narrow = INT_BITS == 32 && FLT_BITS == 32;
    let mut counts = std::collections::BTreeMap::new();
    let mut skipped = Vec::new();
    let mut failures = Vec::new();
    let all = vectors();
    for entry in &all {
        let name = entry.get("name").unwrap().str();
        let mode = entry.get("mode").unwrap().str();
        let widths = entry.get("widths").unwrap().str();
        let applies = match widths {
            "any" => true,
            "64" => wide,
            "32" => narrow,
            other => panic!("widths {:?}", other),
        };
        if !applies {
            skipped.push(format!("{} ({}): widths {}, this build is int{} float{}", name, mode, widths, INT_BITS, FLT_BITS));
            continue;
        }
        let msg = unhex(entry.get("hex").unwrap().str());
        let mut ip = interp();
        let mut fail = |what: String| failures.push(format!("{}: {}", name, what));
        match mode {
            "roundtrip" | "decode" => {
                let expected = match entry.get("expect") {
                    Some(e) => from_expect(&mut ip, e),
                    None => ip.run(entry.get("klong").unwrap().str(), "<vector>").unwrap().unwrap(),
                };
                // the session: header, byte order, decompression, decoding
                match through_session(&msg, &mut ip) {
                    Event::Message { kind, value: Ok(v), size, .. } => {
                        if kind as u8 != msg[1] || size != msg.len() {
                            fail(format!("session reports kind {:?} size {}", kind, size));
                        }
                        if !matches_decoded(&v, &expected, &ip) {
                            fail(format!("session decoded {} but the entry says {}", show(&ip, &v), show(&ip, &expected)));
                        }
                    }
                    Event::Message { value: Err(e), .. } => fail(format!("session: value error {:?}", e)),
                    Event::Failure { error, .. } => fail(format!("session: protocol error {}", error)),
                    _ => fail("session: no message".to_string()),
                }
                // the codec alone, on the bytes after the header (uncompressed entries)
                if msg[2] == 0 {
                    match codec::decode(&msg[8..], msg[0] == 1, &mut ip.syms) {
                        Ok(v) if matches_decoded(&v, &expected, &ip) => {
                            #[cfg(feature = "tables")]
                            if msg[0] == 1 && klong_core::table::Table::from_value(&v).is_some() {
                                let mut out = Vec::new();
                                if codec::encode(&ip.syms, &v, &mut out).is_err() || out != msg[8..] {
                                    fail(format!("table encoded {:02x?}, the entry has {:02x?}", out, &msg[8..]));
                                }
                            }
                        }
                        Ok(v) => fail(format!("decoded {} but the entry says {}", show(&ip, &v), show(&ip, &expected))),
                        Err(e) => fail(format!("decode failed: {:?}", e)),
                    }
                }
                if mode == "roundtrip" {
                    let mut out = Vec::new();
                    match codec::encode(&ip.syms, &expected, &mut out) {
                        Ok(()) if out == msg[8..] => {}
                        Ok(()) => fail(format!("encoded {:02x?}, the entry has {:02x?}", out, &msg[8..])),
                        Err(e) => fail(format!("encode failed: {:?}", e)),
                    }
                    // and the whole message as the session frames it
                    let (mut s, _) = ready_client();
                    let framed = match msg[1] {
                        0 => s.send_async(&ip.syms, &expected, false).unwrap(),
                        1 => s.send_sync(&ip.syms, &expected, true, None, false).unwrap().0,
                        _ => {
                            s.feed(&unhex("010100000a0000006500")).unwrap();
                            assert!(matches!(s.next_event(&mut ip.syms), Some(Event::Message { kind: Kind::Sync, .. })));
                            let mut svc = s.begin_service(Kind::Sync).unwrap();
                            let data = s.send_response(&mut svc, &ip.syms, &expected, false).unwrap();
                            assert!(s.end_service(svc).is_none());
                            data
                        }
                    };
                    if framed != msg {
                        fail(format!("framed {:02x?}", framed));
                    }
                }
            }
            "reject" => {
                assert_eq!(entry.get("error").unwrap().str(), "system", "{}: only protocol errors are in the file", name);
                match through_session(&msg, &mut ip) {
                    Event::Failure { .. } => {}
                    Event::Message { value: Ok(v), .. } => fail(format!("session accepted it as {}", show(&ip, &v))),
                    Event::Message { value: Err(e), .. } => fail(format!("session: a value error ({:?}), not a protocol error", e)),
                    _ => fail("session made no failure (still waiting for bytes)".to_string()),
                }
                if header_consistent(&msg) && msg[2] == 0 {
                    match codec::decode(&msg[8..], msg[0] == 1, &mut ip.syms) {
                        Err(DecodeError::Protocol(_)) => {}
                        Err(DecodeError::Value(e)) => fail(format!("codec: a value error ({:?}), not a protocol error", e)),
                        Ok(v) => fail(format!("codec accepted it as {}", show(&ip, &v))),
                    }
                }
            }
            other => panic!("mode {:?}", other),
        }
        *counts.entry(mode.to_string()).or_insert(0usize) += 1;
    }
    println!("vectors: {} entries, build int{} float{}", all.len(), INT_BITS, FLT_BITS);
    for (mode, n) in &counts {
        println!("  {}: {} run", mode, n);
    }
    println!("  not applicable: {}", skipped.len());
    for s in &skipped {
        println!("    {}", s);
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
    assert!(wide || narrow || !counts.is_empty());
}
