/*
 * klong.h: the C interface of the native Klong runtime (libklong_ffi.a, crate klong-ffi).
 *
 * The library is the whole board runtime: the interpreter, the serial console with its
 * ^A command mode (put, hex, run, ver), a RAM file system, the Klong library and timers.
 * The application supplies a handful of callbacks and calls klong_init once and then
 * klong_run (or klong_step in its own loop) from ONE task. Nothing here is reentrant or
 * thread-safe: every call, and every callback, happens on that task.
 *
 * Only integers and pointers cross this interface: no float or double is passed or
 * returned, so the header means the same under a soft-float and a hard-float C ABI.
 */
#ifndef KLONG_H
#define KLONG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KLONG_ABI_VERSION 1u

/* klong_config.flags */
#define KLONG_FLAG_CRLF 1u /* send CR LF for LF (a terminal on a UART expects it) */
#define KLONG_FLAG_DUMB 2u /* no echo, no line editing, no history (scripts) */
#define KLONG_FLAG_NO_FS 4u /* no RAM file system: put and hex fail, run works */

/* klong_init results */
#define KLONG_OK 0
#define KLONG_ERR_ARG (-1)   /* null config, wrong abi_version or size, a required callback missing */
#define KLONG_ERR_STATE (-2) /* already initialised */

typedef struct klong_config {
    uint32_t abi_version; /* KLONG_ABI_VERSION */
    uint32_t size;        /* sizeof(klong_config) */
    void *ctx;            /* passed back as the first argument of every callback */

    /* Heap. Every allocation of the runtime goes through alloc/free; the runtime never
     * uses more than heap_limit bytes (0: no limit) and keeps heap_reserve of them back so
     * that running out is a `memory` error at the prompt (8 to 16 KiB is right).
     * alloc returns `size` bytes aligned to `align` (a power of two, up to 16 in practice;
     * 8 for 64-bit numbers, so a 4-byte-aligned malloc is not enough) or NULL.
     * free gets the pointer with the size and alignment it was allocated with. */
    void *(*alloc)(void *ctx, size_t size, size_t align);
    void (*free)(void *ctx, void *ptr, size_t size, size_t align);
    size_t heap_limit;
    size_t heap_reserve;

    /* Bytes of native stack the interpreter may use, measured from the frame that calls
     * klong_step or klong_run: the task's stack size minus what the application has used
     * above that call and wants kept for itself (a few KiB). Recursion deeper than this is
     * a `range` error instead of a stack overflow. About 0.8 KiB per level of Klong
     * recursion; 32 KiB is the least that is useful. */
    size_t stack_bytes;

    /* Console. write sends all `len` bytes (it may block for the transmitter).
     * read copies up to `cap` bytes that have ALREADY arrived and returns how many, 0 when
     * there are none; it must not wait. Raw bytes both ways: no echo, no line buffering,
     * no CR/LF translation by the driver. */
    void (*write)(void *ctx, const uint8_t *bytes, size_t len);
    size_t (*read)(void *ctx, uint8_t *buf, size_t cap);

    /* Monotonic milliseconds since some start. Must advance (upload timeouts, timers). */
    uint64_t (*now_ms)(void *ctx);

    /* Optional (may be NULL). idle: called every turn of the main loop and every 4096
     * steps of an evaluation: feed the watchdog, yield; must be cheap.
     * wait: nothing to do; sleep until input may have arrived or `max_ms` have passed
     * (UINT32_MAX: no deadline). Returning early is always right. NULL: busy loop.
     * reset: `.x(code)`; restart the board. If it returns, the program only stops.
     * entropy: 64 random bits to seed `.rn()`. If NULL the clock is used.
     * fatal: the runtime cannot go on (a Rust panic, which is a bug); `msg` is `len`
     * bytes, not terminated. If it returns, the runtime loops forever. */
    void (*idle)(void *ctx);
    void (*wait)(void *ctx, uint32_t max_ms);
    void (*reset)(void *ctx, int32_t code);
    uint64_t (*entropy)(void *ctx);
    void (*fatal)(void *ctx, const char *msg, size_t len);

    const char *name; /* board name for the banner and `ver`, at most 31 bytes; copied */
    uint32_t flags;   /* KLONG_FLAG_* */

    /* Of stack_bytes, what is kept for what runs below the deepest evaluation: a walk over
     * nested data, the lane's pump, a system function. 0: a quarter of stack_bytes (the
     * evaluator gets the rest). A board that has measured what it needs there can give the
     * evaluator more; at least 2048. */
    size_t stack_keep;

    /* Bytes the allocator keeps for each block besides its size (malloc's header: 8 for newlib's).
     * With it every block is counted as its size rounded to 8 plus this, at least 16 (what
     * the arena holds), so heap_limit can be the arena's size less what is not counted;
     * 0: the sizes asked for. */
    size_t heap_overhead;

    /* Blocks are counted rounded up to a multiple of this (the grain the allocator gives them out
     * in: 8); 0: as asked. */
    size_t heap_round;

    /* Optional (may be NULL): resize a block where it is. `ptr` is a block of `alloc` of `old_size` bytes; return 1 if it is now
     * `new_size` bytes at the same address (a larger size needs the room after the block to be free; a smaller one gives the end
     * back), 0 if it cannot be done there, and then nothing has changed: the runtime allocates, copies and frees. A vector that
     * grows by doubling then does not need the old block and the new at once. */
    int (*grow)(void *ctx, void *ptr, size_t old_size, size_t new_size, size_t align);
} klong_config;

/* Build the runtime, run boot.kg if the file system has one, print the banner
 * `Klong <version> (<name>, int64 float64)` and the first prompt. The config is copied. */
int32_t klong_init(const klong_config *config);

/* One turn of the main loop: console input, then timers. Returns 1 when something was
 * done, 0 when there was nothing to do (the caller may then sleep), -1 before klong_init. */
int32_t klong_step(void);

/* The main loop: klong_step, and the wait callback when it had nothing to do. Returns
 * only if klong_init was not called. */
void klong_run(void);

/* A diagnostic of the heap (library built with the cargo feature `heapdiag`; the Pico bench): the board gives a function that
 * fills `out` (room for `cap` integers) for a `mode` and returns how many it wrote (0: nothing for that mode), and the system
 * function `.heap(mode)` returns them as a vector. What the modes mean is the board's (boards/pico-bench/klong_heap.h). The runtime
 * also tells the board's allocator what each allocation is for, in `volatile uint8_t KLONG_ALLOC_TAG` (0 other, 1 vector, 2 string,
 * 3 symbols, 4 globals, 5 parse, 6 ipc, 7 console), set before the `alloc` callback is called. Call once, after klong_init. */
void klong_set_diag(size_t (*diag)(uint32_t mode, int64_t *out, size_t cap));

/* Heap figures of the runtime in bytes: in use, the highest so far, the limit. Any of
 * the pointers may be NULL. */
void klong_heap(size_t *used, size_t *peak, size_t *limit);

/*
 * The lane of a packet switch (library built with the cargo feature `lane`; the Pico bench).
 *
 * The board has one UART lane to a switch that carries only 16-byte frames. The library
 * finds the frames in the byte stream, learns the node number, answers the supervisor's PING
 * and sends STATUS, carries messages of up to 1024 bytes in START/DATA/END/ACK frames, and
 * puts on top of that the kdb+ IPC of the runtime (.cli("node:2"), .srv("node:")) and the
 * console on datagram channel 1. native/boards/pico-bench/README.md says how it is used.
 *
 * Call klong_lane_init once, after klong_init. The two lane callbacks do not wait: lane_read
 * returns the bytes the UART has received (0 if none), lane_write returns how many bytes the
 * transmitter took (0 if it is full; the library offers the rest again). The UART must buffer
 * what arrives while the interpreter is busy: 450 KB/s at 4.5 Mbaud, so an interrupt handler and
 * a receive ring of tens of KB; and the transmitter should have a ring too, drained by interrupt. The library moves frames at every klong_step, every `idle` of the
 * runtime (4096 steps of an evaluation) and inside every IPC call; klong_lane_pump moves
 * them at other times too. Everything on the task that calls klong_init, like the rest.
 */
#define KLONG_LANE_NO_DUTIES 1u  /* do not answer PING or send STATUS (a lane without a supervisor) */
#define KLONG_LANE_NO_CONSOLE 2u /* no console on datagram channel 1 */
#define KLONG_LANE_SELFTEST 4u   /* the boot self-test (see native/boards/pico-bench/README.md) */

typedef struct klong_lane_config {
    uint32_t abi_version; /* KLONG_ABI_VERSION */
    uint32_t size;        /* sizeof(klong_lane_config) */
    void *ctx;            /* first argument of the callbacks */
    size_t (*lane_read)(void *ctx, uint8_t *buf, size_t cap);
    size_t (*lane_write)(void *ctx, const uint8_t *buf, size_t len);
    int32_t node;         /* this node's number 0..16, or -1 to learn it from the first frame sent to it */
    uint32_t seed;        /* random bits: first message id and connection epochs (a restart must not reuse them) */
    uint32_t flags;       /* KLONG_LANE_* */
    uint32_t baud;        /* the lane's baud rate (0: 115200): picks packet size and timeouts */
    const uint32_t *counters; /* NULL, or 4 counters the driver keeps: rx ring overruns, rx FIFO overruns, line errors, tx ring full */
    const char *note;     /* NULL, or why the board restarted (a fault, the watchdog), at most 120 characters: the self-test's reports carry it as ` boot=...` */
} klong_lane_config;

typedef struct klong_lane_stats {
    uint32_t frames_rx, frames_tx, frames_ignored, pings;
    uint32_t bad_frames;   /* frames with the sync bytes and a bad CRC */
    uint32_t skipped_bytes; /* bytes of noise the frame parser threw away */
    uint32_t datagrams_delivered, datagrams_duplicate, datagrams_broken, datagrams_refused, datagrams_resent;
    uint32_t ipc_in, ipc_out, inbox_overflow, console_dropped;
    int32_t console_peer;  /* node attached to the lane console, -1 none */
    uint32_t console_no_peer, console_in_dropped, console_unsolicited; /* output with nobody to send it to (bytes), input dropped (datagrams), channel-1 datagrams without the ACK flag (datagrams) */
    uint32_t uart_rx_ring_overruns, uart_rx_fifo_overruns, uart_line_errors, uart_tx_ring_full; /* the driver's counters */
    int32_t selftest;      /* 0 undecided or off, 1 passed, 2 failed */
} klong_lane_stats;

int32_t klong_lane_init(const klong_lane_config *config);
void klong_lane_pump(void);
/* This node's number, or -1 while it has none. */
int32_t klong_lane_node(void);
void klong_lane_stats_get(klong_lane_stats *out);

#ifdef __cplusplus
}
#endif

#endif /* KLONG_H */
