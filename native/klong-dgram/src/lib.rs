//! klong-dgram: the datagram layer of the Pico bench, `no_std`, no allocation, no unsafe code.
//!
//! * [`crc`]: CRC-16/CCITT-FALSE and CRC-32.
//! * [`frame`]: the 16-byte frame and the sliding-window parser.
//! * [`dgram`]: messages of 1 to 1024 bytes in START, DATA, END and ACK frames, reassembly
//!   per source, duplicate suppression, resends.
//! * [`lane`]: a node on its lane (node number, PING, STATUS) around the datagram layer.
#![no_std]
#![forbid(unsafe_code)]

pub mod crc;
pub mod dgram;
pub mod frame;
pub mod lane;

pub use dgram::{Config, Datagrams, Outcome, SendError, SendResult, MAX_MSG, SOURCES};
pub use frame::{Frame, Parser, FRAME};
pub use lane::{Lane, LaneConfig};

/// How many bytes a node lets wait for its transmitter at `baud`: 12 ms of wire time, in whole
/// frames, at least 4 frames. A deeper queue only delays what must go out behind it (an ACK
/// behind a full window), a shallower one lets the lane idle while the main loop is away.
pub const fn tx_depth(baud: u32) -> usize {
    let bytes = (baud as u64 / 10 * 12 / 1000) as usize;
    let frames = bytes / FRAME;
    (if frames < 4 { 4 } else { frames }) * FRAME
}
