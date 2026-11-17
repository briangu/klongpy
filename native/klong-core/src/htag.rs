//! What an allocation is for, said to the board's allocator before it is made (feature `heap-tags`), so that a diagnostic can
//! tell what kind of object owns each large block of the heap. A scope sets the kind for the allocations made inside it and puts
//! the previous one back; the kind is a number of 0 to 7. Best effort: an allocation made inside a scope that the scope's
//! author did not mean has the scope's kind. With the feature off [`Scope`] is empty and `new` does nothing.

pub const OTHER: u8 = 0;
/// The elements of a vector or list (`try_vec`, `try_push`, `try_extend`).
pub const VECTOR: u8 = 1;
/// The text of a string (`try_string`, `try_str`).
pub const STRING: u8 = 2;
/// The table of symbols: names, the map from names, its growth.
pub const SYMBOLS: u8 = 3;
/// The tables of globals, indexed by symbol.
pub const GLOBALS: u8 = 4;
/// A parse: the tokens, the tree, the copy of the source.
pub const TREE: u8 = 5;
/// The buffers of IPC connections and the event queues.
pub const IPC: u8 = 6;
/// The console's buffers.
pub const CONSOLE: u8 = 7;

#[cfg(feature = "heap-tags")]
mod on {
    use core::sync::atomic::{AtomicU8, Ordering::Relaxed};

    /// Read by the board's allocator (C: `extern volatile uint8_t KLONG_ALLOC_TAG`).
    #[allow(unsafe_code)] // an exported static: the only way to give the allocator the number without a call
    #[no_mangle]
    pub static KLONG_ALLOC_TAG: AtomicU8 = AtomicU8::new(0);

    pub struct Scope(u8);
    impl Scope {
        #[inline(always)]
        pub fn new(kind: u8) -> Scope {
            Scope(KLONG_ALLOC_TAG.swap_relaxed(kind))
        }
    }
    impl Drop for Scope {
        #[inline(always)]
        fn drop(&mut self) {
            KLONG_ALLOC_TAG.store(self.0, Relaxed);
        }
    }
    // load and store only: the target may have no atomic read-modify-write
    trait SwapRelaxed {
        fn swap_relaxed(&self, v: u8) -> u8;
    }
    impl SwapRelaxed for AtomicU8 {
        #[inline(always)]
        fn swap_relaxed(&self, v: u8) -> u8 {
            let old = self.load(Relaxed);
            self.store(v, Relaxed);
            old
        }
    }
}

#[cfg(feature = "heap-tags")]
pub use on::{Scope, KLONG_ALLOC_TAG};

#[cfg(not(feature = "heap-tags"))]
pub struct Scope;
#[cfg(not(feature = "heap-tags"))]
impl Scope {
    #[inline(always)]
    pub fn new(_kind: u8) -> Scope {
        Scope
    }
}
