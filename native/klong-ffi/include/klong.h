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

/* Heap figures of the runtime in bytes: in use, the highest so far, the limit. Any of
 * the pointers may be NULL. */
void klong_heap(size_t *used, size_t *peak, size_t *limit);

#ifdef __cplusplus
}
#endif

#endif /* KLONG_H */
