//! CRC-32 as zlib and Ethernet compute it (reflected polynomial 0xEDB88320, check value
//! 0xCBF43926 for "123456789"). A 16-entry table: 64 bytes of flash, two steps per byte.

const fn table() -> [u32; 16] {
    let mut t = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 4 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

static TABLE: [u32; 16] = table();

#[derive(Clone, Copy)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Crc32 {
        Crc32::new()
    }
}

impl Crc32 {
    pub const fn new() -> Crc32 {
        Crc32(!0)
    }
    pub fn update(&mut self, bytes: &[u8]) {
        let mut c = self.0;
        for &b in bytes {
            c ^= b as u32;
            c = TABLE[(c & 15) as usize] ^ (c >> 4);
            c = TABLE[(c & 15) as usize] ^ (c >> 4);
        }
        self.0 = c;
    }
    pub fn finish(self) -> u32 {
        !self.0
    }
}

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(bytes);
    c.finish()
}
