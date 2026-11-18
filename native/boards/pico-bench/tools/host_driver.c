/*
 * The board's check on the desktop: klong_c.c, the C of the compiled examples and the runtime (libklong_ffi.a, built with the cargo
 * features `cfun` and `heapdiag`-less defaults) in one program. It is what tests/static/test_pico_check.py builds and runs; nothing in it is
 * the firmware's, except that it registers the functions the way main.c does. Lines of Klong are read from standard input, each is typed at
 * the console of the runtime, and its output is printed (the console is dumb: no echo, no prompt, LF). Options: --heap-limit BYTES gives the
 * runtime a heap of that size (the board's pool is about 92,000), --stack BYTES the interpreter's stack.
 */
#define _DARWIN_C_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "klong.h"
#include "klong_c.h"

static char inbuf[1 << 16];
static size_t in_len, in_pos;

static void *h_alloc(void *ctx, size_t size, size_t align) {
    void *p = NULL;
    (void)ctx;
    if (align < sizeof(void *)) align = sizeof(void *);
    return posix_memalign(&p, align, size ? size : 1) ? NULL : p;
}
static void h_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx, (void)size, (void)align;
    free(ptr);
}
static void h_write(void *ctx, const uint8_t *bytes, size_t len) {
    (void)ctx;
    fwrite(bytes, 1, len, stdout);
}
static size_t h_read(void *ctx, uint8_t *buf, size_t cap) {
    (void)ctx;
    size_t n = in_len - in_pos < cap ? in_len - in_pos : cap;
    memcpy(buf, inbuf + in_pos, n);
    in_pos += n;
    return n;
}
static uint64_t h_now_ms(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u;
}
static uint64_t now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
}
static void h_idle(void *ctx) { (void)ctx; }
static void h_wait(void *ctx, uint32_t ms) { (void)ctx, (void)ms; }
static void h_reset(void *ctx, int32_t code) { (void)ctx, (void)code; }

int main(int argc, char **argv) {
    size_t heap = 64u << 20, stack = 256u << 10;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--heap-limit")) heap = (size_t)strtoull(argv[i + 1], NULL, 0);
        else if (!strcmp(argv[i], "--stack")) stack = (size_t)strtoull(argv[i + 1], NULL, 0);
        else return 2;
    }
    in_len = fread(inbuf, 1, sizeof inbuf - 1, stdin);
    klong_config cfg = {
        .abi_version = KLONG_ABI_VERSION, .size = sizeof cfg, .alloc = h_alloc, .free = h_free,
        .heap_limit = heap, .heap_reserve = heap > (1u << 20) ? 16u << 10 : 6u << 10, .stack_bytes = stack,
        .write = h_write, .read = h_read, .now_ms = h_now_ms, .idle = h_idle, .wait = h_wait, .reset = h_reset,
        .name = "host-check", .flags = KLONG_FLAG_DUMB,
    };
    if (klong_init(&cfg) != KLONG_OK) return 3;
    if (klong_c_register(now_us) != KLONG_OK) return 4;
    for (int quiet = 0; quiet < 3;) {
        if (klong_step() == 1) quiet = 0; else quiet++;
    }
    fflush(stdout);
    return 0;
}
