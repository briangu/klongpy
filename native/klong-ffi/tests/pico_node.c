/*
 * One node of the Pico bench as a host process: the runtime of klong-ffi with the lane and the boot self-test, the firmware's
 * own allocator (boards/pico-bench/klong_heap.h) over a model of newlib's malloc with an arena of a given size, the
 * firmware's heap limits, and a lane that is two file descriptors: 3 (frames from the switch, non-blocking) and 4 (frames
 * to it). tests/test_native_pico_pair.py is the switch and the supervisor: it starts two of these (nodes 2 and 3 of the
 * bench: the higher one runs the self-test as a client) and watches what the supervisor sees.
 *
 *   pico_node ARENA_BYTES [old]
 *
 * `old` is the policy of the image that failed on the boards: sizes counted as asked (no header), the limit 4% below the
 * arena, and a NULL from malloc is the SDK's panic("Out of memory"): the process prints PANIC and stops answering, as a
 * node spinning in panic() does. Without it a NULL is an answer, as in the fixed firmware (`__real_malloc`), and what
 * cannot fail and does ends in `fatal`: the process prints FATAL and exits, as the board reboots.
 */
#define _DARWIN_C_SOURCE
#define _DEFAULT_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "klong.h"
#include "klong_heap_arena.h"

static int old_policy;
static void *panic_malloc(size_t n) {
    void *p = mock_malloc(n);
    if (!p && old_policy) {
        fprintf(stderr, "PANIC Out of memory (malloc %zu, arena %zu)\n", n, mock_arena_bytes);
        for (;;) pause();
    }
    return p;
}
#define KH_MALLOC panic_malloc
#define KH_FREE mock_free
#define KH_MEMALIGN mock_memalign
#define KH_POISON 1
#include "klong_heap.h"
#include "klong_heap_arena_impl.h"

static void h_write(void *ctx, const uint8_t *b, size_t n) { (void)ctx, (void)b, (void)n; }
static size_t h_read(void *ctx, uint8_t *b, size_t cap) { (void)ctx, (void)b, (void)cap; return 0; }
static uint64_t h_now_ms(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u;
}
static void h_fatal(void *ctx, const char *msg, size_t len) {
    (void)ctx;
    fprintf(stderr, "FATAL %.*s\n", (int)len, msg);
    _exit(98);
}
static size_t l_read(void *ctx, uint8_t *b, size_t cap) {
    (void)ctx;
    ssize_t n = read(3, b, cap);
    return n > 0 ? (size_t)n : 0;
}
static size_t l_write(void *ctx, const uint8_t *b, size_t n) {
    (void)ctx;
    ssize_t w = write(4, b, n);
    return w > 0 ? (size_t)w : 0;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    size_t arena = (size_t)strtoul(argv[1], NULL, 10);
    old_policy = argc > 2 && strcmp(argv[2], "old") == 0;
    if (arena > KH_TESTED_ARENA) return 2;
    mock_arena_bytes = arena;
    mock_init();
    fcntl(3, F_SETFL, O_NONBLOCK);
    klong_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = kh_alloc;
    cfg.free = kh_free;
    if (old_policy) {
        size_t slack = arena / 25;
        cfg.heap_limit = arena - (slack < 2048 ? 2048 : slack);
        cfg.heap_reserve = 6 * 1024;
        cfg.heap_overhead = 0;
    } else {
        kh_limits(arena, &cfg.heap_limit, &cfg.heap_reserve);
        cfg.heap_overhead = 8;
    }
    cfg.stack_bytes = 4u << 20;
    cfg.write = h_write;
    cfg.read = h_read;
    cfg.now_ms = h_now_ms;
    cfg.fatal = h_fatal;
    cfg.name = "pico-node";
    cfg.flags = KLONG_FLAG_CRLF;
    if (klong_init(&cfg) != KLONG_OK) return 3;
    klong_lane_config lane;
    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = l_read;
    lane.lane_write = l_write;
    lane.node = -1;
    lane.seed = (uint32_t)getpid();
    lane.flags = KLONG_LANE_SELFTEST;
    lane.baud = 4500000;
    lane.note = getenv("PICO_NOTE");
    if (klong_lane_init(&lane) != KLONG_OK) return 4;
    for (;;) {
        if (klong_step() != 1) usleep(200);
    }
}
