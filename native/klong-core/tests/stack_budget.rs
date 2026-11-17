//! The limits a board derives from its stack: `Config::for_stack` keeps a quarter of it for what runs below the deepest
//! evaluation, `for_stack_keep` what the board says (the Pico bench: 16 KB of 133 KB).
use klong_core::Config;

#[test]
fn a_quarter_is_kept_by_default() {
    let c = Config::for_stack(90 * 1024);
    assert_eq!(c.stack_limit, Some(90 * 1024 - 22 * 1024 - 512));
    assert_eq!(Config::for_stack_keep(90 * 1024, 0).stack_limit, c.stack_limit);
    assert_eq!(c.max_nest, ((22 * 1024 + 512) / 2 / klong_core::interp::STACK_PER_NEST) as u32);
}

#[test]
fn a_board_that_knows_what_it_needs_keeps_that() {
    let bytes = 133 * 1024;
    let c = Config::for_stack_keep(bytes, 16 * 1024);
    assert_eq!(c.stack_limit, Some(bytes - 16 * 1024));
    assert_eq!(c.max_nest, 21); // (16 KB / 2) / 384 bytes
    // never less than 2 KB, never more than half
    assert_eq!(Config::for_stack_keep(bytes, 100).stack_limit, Some(bytes - 2048));
    assert_eq!(Config::for_stack_keep(bytes, bytes).stack_limit, Some(bytes - bytes / 2));
}
