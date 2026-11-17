//! Serial framing (`docs/native/index.md`): a raw byte pipe with no boundaries and no
//! error detection (a UART, RS-485, a radio modem in transparent mode) carries packets as
//!
//! ```text
//! frame = COBS( payload || CRC-16/CCITT-FALSE(payload) ) || 0x00      CRC high byte first
//! ```
//!
//! A frame whose COBS decoding or CRC fails is dropped silently; the reliability layer
//! ([`crate::rel`]) retransmits. No I/O here: [`encode_frame`] gives the bytes to write and
//! [`Deframer`] takes the bytes read.
use alloc::vec::Vec;

/// Default payload bytes per frame (the frame is then at most 245 bytes on the wire).
pub const DEFAULT_MTU: usize = 240;

/// CRC-16/CCITT-FALSE (CRC-16/IBM-3740): polynomial 0x1021, initial value 0xFFFF, no
/// reflection, no final XOR. Of `123456789` it is 0x29B1.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xffff;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

/// Bytes COBS needs for `n` bytes of data, without the delimiter.
pub fn cobs_max(n: usize) -> usize {
    n + n / 254 + 1
}

/// COBS-encode `src`, appending to `out` (no delimiter). False when `out` cannot grow.
/// A block of 254 non-zero bytes that ends the data is not followed by a `01`.
pub fn cobs_encode(src: &[u8], out: &mut Vec<u8>) -> bool {
    if out.try_reserve(cobs_max(src.len())).is_err() {
        return false;
    }
    let mut code_at = out.len();
    out.push(0);
    let mut code: u8 = 1;
    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        i += 1;
        if b == 0 {
            out[code_at] = code;
            code_at = out.len();
            out.push(0);
            code = 1;
        } else {
            out.push(b);
            code += 1;
            if code == 0xff {
                out[code_at] = code;
                if i == src.len() {
                    return true; // a full block ends the data: no trailing 01
                }
                code_at = out.len();
                out.push(0);
                code = 1;
            }
        }
    }
    out[code_at] = code;
    true
}

/// COBS-decode `src` (without its delimiter), appending to `out`. False for a malformed
/// frame (a zero inside it, a block longer than what is left) or when `out` cannot grow.
/// Both encodings of a final full block (with and without the trailing `01`) are accepted.
pub fn cobs_decode(src: &[u8], out: &mut Vec<u8>) -> bool {
    if out.try_reserve(src.len()).is_err() {
        return false;
    }
    let mut i = 0;
    while i < src.len() {
        let code = src[i] as usize;
        if code == 0 || i + code > src.len() {
            return false;
        }
        let block = &src[i + 1..i + code];
        if block.contains(&0) {
            return false;
        }
        out.extend_from_slice(block);
        i += code;
        if code != 0xff && i < src.len() {
            out.push(0);
        }
    }
    true
}

/// The bytes on the wire for one packet, appended to `out`. False when `out` cannot grow.
pub fn encode_frame(payload: &[u8], out: &mut Vec<u8>, scratch: &mut Vec<u8>) -> bool {
    scratch.clear();
    if scratch.try_reserve(payload.len() + 2).is_err() {
        return false;
    }
    scratch.extend_from_slice(payload);
    scratch.extend_from_slice(&crc16(payload).to_be_bytes());
    if !cobs_encode(scratch, out) || out.try_reserve(1).is_err() {
        return false;
    }
    out.push(0);
    true
}

/// Takes the bytes read from the pipe and gives back whole, checked packets.
pub struct Deframer {
    /// Bytes of the frame being received (COBS-encoded).
    raw: Vec<u8>,
    /// The frame is longer than any valid one: skip to the next delimiter.
    overrun: bool,
    max_raw: usize,
    /// The packet last returned.
    packet: Vec<u8>,
    /// Frames dropped: bad COBS, bad CRC, too long.
    pub dropped: u32,
}

impl Deframer {
    /// `mtu` is the largest payload of a frame.
    pub fn new(mtu: usize) -> Deframer {
        Deframer { raw: Vec::new(), overrun: false, max_raw: cobs_max(mtu + 2), packet: Vec::new(), dropped: 0 }
    }

    /// Take bytes from `data` up to and including the next frame delimiter. Returns how many
    /// bytes were consumed and, when they completed a valid frame, its payload. Call again
    /// with the rest of `data`.
    pub fn push<'a>(&'a mut self, data: &[u8]) -> (usize, Option<&'a [u8]>) {
        for (i, &b) in data.iter().enumerate() {
            if b != 0 {
                if self.overrun {
                    continue;
                }
                if self.raw.len() >= self.max_raw || self.raw.try_reserve(1).is_err() {
                    self.overrun = true;
                    self.raw.clear();
                    continue;
                }
                self.raw.push(b);
                continue;
            }
            // a delimiter
            let overrun = core::mem::take(&mut self.overrun);
            if overrun {
                self.dropped += 1;
                continue;
            }
            if self.raw.is_empty() {
                continue; // an idle line, or a delimiter sent first to flush noise
            }
            self.packet.clear();
            let ok = cobs_decode(&self.raw, &mut self.packet);
            self.raw.clear();
            let n = self.packet.len();
            if ok && n >= 2 && crc16(&self.packet[..n - 2]) == u16::from_be_bytes([self.packet[n - 2], self.packet[n - 1]]) {
                self.packet.truncate(n - 2);
                return (i + 1, Some(&self.packet));
            }
            self.dropped += 1;
        }
        (data.len(), None)
    }
}
