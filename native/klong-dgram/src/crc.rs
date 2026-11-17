//! The two checksums of the frame format. Bitwise or nibble-table: small, no big tables in
//! flash, and fast enough for 2.9 KB/s.

/// CRC-16/CCITT-FALSE: polynomial 0x1021, init 0xFFFF, no reflection, no final XOR.
/// `crc16(b"123456789") == 0x29B1`.
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xffff;
    for &b in data {
        crc = (crc << 8) ^ CRC16_TABLE[((crc >> 8) as u8 ^ b) as usize];
    }
    crc
}

/// One table of 512 bytes: at 4.5 Mbaud a frame arrives every 36 us, and the bitwise loop
/// (96 shifts) would take a Cortex-M0+ most of that.
const CRC16_TABLE: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = (i as u16) << 8;
        let mut k = 0;
        while k < 8 {
            c = if c & 0x8000 != 0 { (c << 1) ^ 0x1021 } else { c << 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

const NIBBLE: [u32; 16] = [
    0x0000_0000, 0x1db7_1064, 0x3b6e_20c8, 0x26d9_30ac, 0x76dc_4190, 0x6b6b_51f4, 0x4db2_6158, 0x5005_713c, 0xedb8_8320, 0xf00f_9344, 0xd6d6_a3e8, 0xcb61_b38c, 0x9b64_c2b0, 0x86d3_d2d4, 0xa00a_e278, 0xbdbd_f21c,
];

/// An incremental CRC-32 (zlib, IEEE 802.3, reflected). `crc32(b"123456789") == 0xCBF43926`.
#[derive(Clone, Copy)]
pub struct Crc32(u32);

impl Crc32 {
    pub const fn new() -> Crc32 {
        Crc32(0xffff_ffff)
    }
    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        for &b in data {
            c ^= b as u32;
            c = (c >> 4) ^ NIBBLE[(c & 15) as usize];
            c = (c >> 4) ^ NIBBLE[(c & 15) as usize];
        }
        self.0 = c;
    }
    pub fn finish(&self) -> u32 {
        !self.0
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Crc32::new()
    }
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}
