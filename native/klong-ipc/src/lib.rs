//! klong-ipc: the kdb+ IPC wire protocol and the links under it, without any I/O.
//!
//! `no_std` + `alloc`, no dependency beyond `klong-core`. Bytes in, bytes and events out: a
//! transport (TCP, a UART, a radio) only moves the bytes. Design: `docs/ipc_capabilities.md`
//! (the protocol shared with KlongPy, whose `klongpy/ipc` is the executable reference) and
//! `docs/native/index.md` (links).
//!
//! * [`codec`]: one kdb+ object to and from a [`klong_core::Value`].
//! * [`compress`]: message decompression (always) and compression (feature `compress`).
//! * [`session`]: handshake, framing, sync/async/response, levelled response matching,
//!   deadlines against a clock the caller reads.
//! * [`rel`]: the packet reliability layer (1-byte header, go-back-N) that turns bounded,
//!   lossy packets into the ordered byte stream the session needs.
//! * [`serial`]: COBS + CRC-16 framing that turns a raw byte pipe into packets.
//!
//! Nothing here panics on any input: lengths are checked before allocating, allocation is
//! fallible (`try_reserve`), arithmetic on sizes from the wire is checked.
#![no_std]
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

pub mod codec;
pub mod compress;
pub mod rel;
pub mod serial;
pub mod session;

pub use codec::{decode, encode, encode_call, DecodeError, KdbError, KdbLambda, KdbOpaque};
pub use session::{Event, Kind, Pending, PendingId, Session};
