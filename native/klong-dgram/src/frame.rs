//! The 16-byte frame of the switch, and the parser that finds frames in a byte stream.
//!
//! ```text
//! [0..1] A5 5A  [2] 01  [3] type  [4] source  [5] destination  [6..7] sequence u16 LE
//! [8] sub-type  [9] payload length 0..4  [10..13] payload, unused bytes 0
//! [14..15] CRC-16/CCITT-FALSE over bytes 2..13, LE
//! ```
use crate::crc::crc16;

pub const FRAME: usize = 16;
pub const SUPERVISOR: u8 = 16;
pub const FPGA: u8 = 17;
pub const BROADCAST: u8 = 254;

pub const T_PING: u8 = 1;
pub const T_PONG: u8 = 2;
pub const T_STATUS: u8 = 3;
pub const T_APP_DATA: u8 = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Frame {
    pub ftype: u8,
    pub src: u8,
    pub dst: u8,
    pub seq: u16,
    /// Byte 8: the sub-type of APP_DATA ("ttl" in the switch's own sources).
    pub sub: u8,
    pub len: u8,
    pub payload: [u8; 4],
}

fn is_core(a: u8) -> bool {
    (64..80).contains(&a)
}
/// Fan-out destinations reach every port.
pub fn is_fanout(dst: u8) -> bool {
    dst == BROADCAST || (32..48).contains(&dst)
}

impl Frame {
    pub fn new(ftype: u8, src: u8, dst: u8, seq: u16, sub: u8, data: &[u8]) -> Frame {
        let mut payload = [0u8; 4];
        let n = data.len().min(4);
        payload[..n].copy_from_slice(&data[..n]);
        Frame { ftype, src, dst, seq, sub, len: n as u8, payload }
    }

    /// What the switch accepts: a known type, addresses it routes, length 0..4, unused payload bytes 0.
    pub fn valid(&self) -> bool {
        (T_PING..=T_APP_DATA).contains(&self.ftype)
            && (self.src <= FPGA || is_core(self.src))
            && (self.dst <= FPGA || is_fanout(self.dst) || is_core(self.dst))
            && self.len <= 4
            && self.payload[self.len as usize..].iter().all(|&b| b == 0)
    }

    pub fn data(&self) -> &[u8] {
        &self.payload[..self.len as usize]
    }

    /// The 16 wire bytes; `None` if the frame is not valid.
    pub fn encode(&self) -> Option<[u8; FRAME]> {
        if !self.valid() {
            return None;
        }
        let mut o = [0u8; FRAME];
        o[0] = 0xa5;
        o[1] = 0x5a;
        o[2] = 1;
        o[3] = self.ftype;
        o[4] = self.src;
        o[5] = self.dst;
        o[6] = self.seq as u8;
        o[7] = (self.seq >> 8) as u8;
        o[8] = self.sub;
        o[9] = self.len;
        o[10..14].copy_from_slice(&self.payload);
        let c = crc16(&o[2..14]);
        o[14] = c as u8;
        o[15] = (c >> 8) as u8;
        Some(o)
    }

    pub fn decode(b: &[u8; FRAME]) -> Option<Frame> {
        if b[0] != 0xa5 || b[1] != 0x5a || b[2] != 1 || crc16(&b[2..14]) != (b[14] as u16 | (b[15] as u16) << 8) {
            return None;
        }
        let f = Frame { ftype: b[3], src: b[4], dst: b[5], seq: b[6] as u16 | (b[7] as u16) << 8, sub: b[8], len: b[9], payload: [b[10], b[11], b[12], b[13]] };
        if f.valid() {
            Some(f)
        } else {
            None
        }
    }
}

/// A 16-byte sliding window over the byte stream: a frame is found wherever `A5 5A 01` and a
/// good CRC line up, whatever came before (noise, a torn frame, lost bytes). Bytes that start
/// no frame are dropped one at a time, so a frame that begins inside a damaged one is still seen.
pub struct Parser {
    buf: [u8; FRAME],
    len: usize,
    /// Bytes thrown away while looking for a frame.
    pub skipped: u32,
    /// Windows that had the sync bytes and failed the CRC or validity.
    pub bad: u32,
}

const HEAD: [u8; 3] = [0xa5, 0x5a, 1];

impl Parser {
    pub const fn new() -> Parser {
        Parser { buf: [0; FRAME], len: 0, skipped: 0, bad: 0 }
    }
    pub fn reset(&mut self) {
        self.len = 0;
    }
    fn drop_first(&mut self) {
        self.buf.copy_within(1..self.len, 0);
        self.len -= 1;
        self.skipped += 1;
    }
    /// Feed one byte; a frame when it completes one.
    pub fn push(&mut self, byte: u8) -> Option<Frame> {
        self.buf[self.len] = byte;
        self.len += 1;
        loop {
            if self.len == 0 {
                return None;
            }
            let n = self.len.min(3);
            if self.buf[..n] != HEAD[..n] {
                self.drop_first();
                continue;
            }
            if self.len < FRAME {
                return None;
            }
            match Frame::decode(&self.buf) {
                Some(f) => {
                    self.len = 0;
                    return Some(f);
                }
                None => {
                    self.bad += 1;
                    self.drop_first();
                }
            }
        }
    }
}

impl Default for Parser {
    fn default() -> Self {
        Parser::new()
    }
}
