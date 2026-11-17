//! Intel HEX, one record at a time (docs/native/porting.md). No allocation: a record is
//! decoded into a caller's 255-byte buffer.
//!
//! `:` count(1) address(2, big-endian) type(1) data(count) checksum(1), all as pairs of hex
//! digits; the bytes of a record sum to zero modulo 256.

/// The most data bytes a record can carry.
pub const MAX_DATA: usize = 255;
/// The longest record line: the colon and (1 + 2 + 1 + 255 + 1) bytes as hex pairs.
pub const MAX_LINE: usize = 1 + 2 * (5 + MAX_DATA);

#[derive(Debug, PartialEq, Eq)]
pub enum Record<'a> {
    /// Type 00: `data` at the 16-bit address, to be added to the current base.
    Data { addr: u16, data: &'a [u8] },
    /// Type 01.
    Eof,
    /// Type 02: the base becomes `segment << 4`.
    ExtSegment(u16),
    /// Type 04: the base becomes `upper << 16`.
    ExtLinear(u16),
    /// Types 03 and 05, start addresses: common tools emit them; they mean nothing to a file.
    Start,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HexError {
    /// The line does not start with a colon.
    NoColon,
    /// Something that is not a hex digit, or an odd number of digits.
    Digit,
    /// Fewer than the five bytes every record has.
    Short,
    /// The byte count does not agree with the length of the line.
    Length,
    Checksum,
    /// A record type other than 00 to 05.
    Type(u8),
    /// A record whose fixed-size payload has the wrong size (02 and 04 carry 2 bytes, 03 and
    /// 05 carry 4, 01 none).
    Field,
}

impl HexError {
    pub fn text(self) -> &'static str {
        match self {
            HexError::NoColon => "no colon",
            HexError::Digit => "not hex",
            HexError::Short => "short record",
            HexError::Length => "length mismatch",
            HexError::Checksum => "checksum",
            HexError::Type(_) => "unsupported record type",
            HexError::Field => "bad field size",
        }
    }
}

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// Decode one line (without its line end; blanks around it are ignored).
pub fn parse<'a>(line: &[u8], buf: &'a mut [u8; MAX_DATA]) -> Result<Record<'a>, HexError> {
    let mut line = line;
    while let [b' ' | b'\t' | b'\r', rest @ ..] = line {
        line = rest;
    }
    while let [rest @ .., b' ' | b'\t' | b'\r'] = line {
        line = rest;
    }
    let digits = match line {
        [b':', rest @ ..] => rest,
        _ => return Err(HexError::NoColon),
    };
    if digits.len() % 2 != 0 {
        return Err(HexError::Digit);
    }
    let mut head = [0u8; 4];
    let mut sum = 0u8;
    let mut n = 0usize;
    let mut last = 0u8;
    for pair in digits.chunks_exact(2) {
        let b = match (nibble(pair[0]), nibble(pair[1])) {
            (Some(h), Some(l)) => h << 4 | l,
            _ => return Err(HexError::Digit),
        };
        sum = sum.wrapping_add(b);
        if n < 4 {
            head[n] = b;
        } else {
            // the byte before this one was data, not the checksum
            if n > 4 {
                let at = n - 5;
                if at >= MAX_DATA {
                    return Err(HexError::Length);
                }
                buf[at] = last;
            }
            last = b;
        }
        n += 1;
    }
    if n < 5 {
        return Err(HexError::Short);
    }
    let count = head[0] as usize;
    if n != count + 5 {
        return Err(HexError::Length);
    }
    if sum != 0 {
        return Err(HexError::Checksum);
    }
    let addr = u16::from_be_bytes([head[1], head[2]]);
    let data = &buf[..count];
    match head[3] {
        0 => Ok(Record::Data { addr, data }),
        1 if count == 0 => Ok(Record::Eof),
        2 if count == 2 => Ok(Record::ExtSegment(u16::from_be_bytes([data[0], data[1]]))),
        4 if count == 2 => Ok(Record::ExtLinear(u16::from_be_bytes([data[0], data[1]]))),
        3 | 5 if count == 4 => Ok(Record::Start),
        1..=5 => Err(HexError::Field),
        t => Err(HexError::Type(t)),
    }
}

/// Write one record as text (no line end) into `out`; the number of bytes written.
/// `out` must hold `11 + 2 * data.len()` bytes and `data` at most 255.
pub fn write(kind: u8, addr: u16, data: &[u8], out: &mut [u8]) -> usize {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let count = data.len().min(MAX_DATA);
    let need = 11 + 2 * count;
    if out.len() < need {
        return 0;
    }
    out[0] = b':';
    let mut at = 1;
    let mut sum = 0u8;
    let head = [count as u8, (addr >> 8) as u8, addr as u8, kind];
    for &b in head.iter().chain(data[..count].iter()) {
        out[at] = HEX[(b >> 4) as usize];
        out[at + 1] = HEX[(b & 15) as usize];
        at += 2;
        sum = sum.wrapping_add(b);
    }
    let cs = sum.wrapping_neg();
    out[at] = HEX[(cs >> 4) as usize];
    out[at + 1] = HEX[(cs & 15) as usize];
    need
}
