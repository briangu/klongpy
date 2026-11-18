//! klong-core: the Klong language without a board, an operating system or a network.
//!
//! `no_std` + `alloc`. Everything outside pure computation reaches the core through the
//! embedder: [`Hooks`] (the periodic tick that carries interrupts) and the system functions an
//! embedder registers with [`Interp::register`]. Design: `docs/native/index.md`.
#![no_std]
// (`link_section` counts as unsafe code for the lint; the feature `ram-hot` uses it on a few functions, `heap-tags` exports one
// static to the board's allocator, and nothing else is unsafe)
#![cfg_attr(not(any(feature = "ram-hot", feature = "heap-tags")), forbid(unsafe_code))]
#![cfg_attr(any(feature = "ram-hot", feature = "heap-tags"), deny(unsafe_code))]
#![deny(clippy::unwrap_used, clippy::expect_used)]

extern crate alloc;

#[cfg(feature = "bigint")]
pub mod big;
pub mod dump;
pub mod error;
pub mod fmt;
pub mod htag;
pub mod interp;
pub mod lex;
pub mod ops;
pub mod parse;
pub mod sym;
pub mod tree;
pub mod value;

mod adverbs;

pub use error::{Error, Kind, Source, Span};
pub use interp::{Config, Hooks, Interp, NoHooks};
pub use sym::SymId;
pub use value::{Flt, Int, Value};

/// Width in bits of the integer and float types of this build.
pub const INT_BITS: u32 = Int::BITS;
#[cfg(not(feature = "float32"))]
pub const FLT_BITS: u32 = 64;
#[cfg(feature = "float32")]
pub const FLT_BITS: u32 = 32;

// The size budget of the design (docs/native/index.md): 16 bytes on 64-bit targets and at
// most 16 on 32-bit ones. Checked when the crate is compiled, on every target.
const _: () = assert!(core::mem::size_of::<Value>() <= 16);
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<Value>() == 16);
#[cfg(all(target_pointer_width = "32", feature = "int32", feature = "float32"))]
const _: () = assert!(core::mem::size_of::<Value>() == 8);
