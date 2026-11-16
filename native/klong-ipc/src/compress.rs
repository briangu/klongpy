//! kdb+ IPC message compression (`ref-kdb-ipc.md` section 5), ports of `klongpy/ipc/compress.py`
//! (themselves ports of the published Java and JavaScript client code).
//!
//! Both functions work on whole messages including the 8-byte header.
//!
//! [`decompress`] is always built: a peer that was offered capability 3 may compress.
//! [`compress`] needs the feature `compress` (design section 10: small devices leave it out).
//!
//! Safety of [`decompress`]: the declared size is checked against `max_size` and against what
//! the stream could possibly expand to (125 times its length) before anything is allocated,
//! the allocation is fallible, and every back-reference is validated as the Java client does.
use alloc::string::String;
use alloc::vec::Vec;

/// kdb+ compresses only messages larger than this.
pub const COMPRESS_THRESHOLD: usize = 2000;

/// A full group (1 flag byte, 8 back-references of 2 bytes) of 17 input bytes produces
/// 8 * 257 = 2056 output bytes, a ratio of about 121.
const MAX_EXPANSION: usize = 125;

pub enum Failure {
    /// Not a valid compressed message (a protocol error).
    Bad(&'static str),
    /// No memory for the uncompressed message.
    Memory,
}

fn u32_at(data: &[u8], at: usize, little: bool) -> u32 {
    let a = [data[at], data[at + 1], data[at + 2], data[at + 3]];
    if little {
        u32::from_le_bytes(a)
    } else {
        u32::from_be_bytes(a)
    }
}

fn put_u32(data: &mut [u8], at: usize, little: bool, v: u32) {
    let b = if little { v.to_le_bytes() } else { v.to_be_bytes() };
    data[at..at + 4].copy_from_slice(&b);
}

/// The uncompressed message for the compressed message `data`.
pub fn decompress(data: &[u8], max_size: usize) -> Result<Vec<u8>, Failure> {
    let n = data.len();
    if n < 12 {
        return Err(Failure::Bad("compressed message shorter than its 12-byte header"));
    }
    if data[0] > 1 {
        return Err(Failure::Bad("bad byte-order flag in message header"));
    }
    let little = data[0] == 1;
    let total = u32_at(data, 4, little) as usize;
    let usize_ = u32_at(data, 8, little) as usize;
    if total != n {
        return Err(Failure::Bad("compressed length field does not match the message"));
    }
    if usize_ < 8 || usize_ > max_size || usize_ > 0x7fff_ffff {
        return Err(Failure::Bad("declared uncompressed size out of range"));
    }
    if usize_ - 8 > MAX_EXPANSION.saturating_mul(n - 12) {
        return Err(Failure::Bad("declared uncompressed size impossible for the compressed length"));
    }
    let mut dst: Vec<u8> = Vec::new();
    dst.try_reserve_exact(usize_).map_err(|_| Failure::Memory)?;
    dst.resize(usize_, 0);
    let mut aa = [0usize; 256];
    let mut s = 8usize; // write position in dst
    let mut p = 8usize; // next position whose pair is not hashed yet
    let mut d = 12usize; // read position in data
    let mut i = 0u32; // current flag bit, 0 = need a new flag byte
    let mut f = 0u32;
    const TRUNC: Failure = Failure::Bad("compressed stream truncated");
    while s < usize_ {
        if i == 0 {
            if d >= n {
                return Err(TRUNC);
            }
            f = data[d] as u32;
            d += 1;
            i = 1;
        }
        let backref = f & i != 0;
        let mut ext = 0usize;
        if backref {
            if d + 2 > n {
                return Err(TRUNC);
            }
            let mut r = aa[data[d] as usize];
            ext = data[d + 1] as usize;
            d += 2;
            // (s >= 8 always, so s - 1 does not wrap)
            if r < 8 || r >= s - 1 || ext + 2 > usize_ - s {
                return Err(Failure::Bad("invalid back-reference in compressed stream"));
            }
            dst[s] = dst[r];
            dst[s + 1] = dst[r + 1];
            s += 2;
            r += 2;
            // forward byte-by-byte copy; the source may overlap the destination
            for m in 0..ext {
                dst[s + m] = dst[r + m];
            }
        } else {
            if d >= n {
                return Err(TRUNC);
            }
            dst[s] = data[d];
            s += 1;
            d += 1;
        }
        // index every adjacent pair written so far (before skipping the run)
        while p + 1 < s {
            aa[(dst[p] ^ dst[p + 1]) as usize] = p;
            p += 1;
        }
        if backref {
            s += ext;
            p = s;
        }
        i <<= 1;
        if i == 256 {
            i = 0;
        }
    }
    dst[0] = data[0];
    dst[1] = data[1];
    dst[2] = 0;
    dst[3] = data[3];
    put_u32(&mut dst, 4, little, usize_ as u32);
    Ok(dst)
}

/// The text of a failure, for a protocol error message.
pub fn failure_text(f: &Failure) -> String {
    String::from(match f {
        Failure::Bad(s) => s,
        Failure::Memory => "no memory for the uncompressed message",
    })
}

/// `message` (whole, uncompressed, with its header) compressed, or `None` when kdb+'s rules
/// say to send it as it is: not longer than `threshold` bytes (2000), or the compressed form
/// would not be under half of it, or there is no memory for the attempt.
#[cfg(feature = "compress")]
pub fn compress(y: &[u8], threshold: usize) -> Option<Vec<u8>> {
    let t = y.len();
    if t <= threshold || t < 20 || y[0] > 1 || y[2] != 0 || t > 0x7fff_ffff {
        return None;
    }
    let little = y[0] == 1;
    let e = t / 2;
    let mut out: Vec<u8> = Vec::new();
    out.try_reserve_exact(e).ok()?;
    out.resize(e, 0);
    if e < 30 {
        return None;
    }
    out[0..4].copy_from_slice(&y[0..4]);
    out[2] = 1;
    put_u32(&mut out, 8, little, t as u32);
    let mut c = 12usize;
    let mut d = 12usize;
    let mut f = 0u8;
    let mut a = [0usize; 256];
    let mut i = 0u8;
    let mut s = 8usize;
    let mut s0 = 0usize;
    let mut h0 = 0usize;
    let mut h = 0usize;
    while s < t {
        if i == 0 {
            if d + 17 > e {
                return None;
            }
            i = 1;
            out[c] = f;
            c = d;
            d += 1;
            f = 0;
        }
        let mut literal = s + 3 > t;
        let mut p = 0usize;
        if !literal {
            h = (y[s] ^ y[s + 1]) as usize;
            p = a[h];
            literal = p == 0 || y[s] != y[p];
        }
        if s0 > 0 {
            a[h0] = s0;
            s0 = 0;
        }
        if literal {
            h0 = h;
            s0 = s;
            out[d] = y[s];
            d += 1;
            s += 1;
        } else {
            a[h] = s;
            f |= i;
            p += 2;
            s += 2;
            let r = s;
            let q = (s + 255).min(t);
            while s < t && y[p] == y[s] {
                s += 1;
                if s >= q {
                    break;
                }
                p += 1;
            }
            out[d] = h as u8;
            d += 1;
            out[d] = (s - r) as u8;
            d += 1;
        }
        i = i.wrapping_mul(2);
    }
    out[c] = f;
    put_u32(&mut out, 4, little, d as u32);
    out.truncate(d);
    Some(out)
}
