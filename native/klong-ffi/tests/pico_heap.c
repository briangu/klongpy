/*
 * The Pico bench firmware's heap on the host: boards/pico-bench/klong_heap.h (the very source main.c compiles:
 * the size-class allocator, its slab table and trim, and the numbers the runtime's heap limit is made of) over a
 * model of newlib's malloc (first fit, 8-byte headers, merging neighbours on free) with an arena of the size
 * the linked image has (KH_TESTED_ARENA), under the runtime of klong-ffi.
 *
 * It checks what the owner found on the boards: that the largest vector is not far below the heap (a producer of
 * known length such as `!n` allocates once, at its final size: what fits is the limit less the reserve less what
 * is in use, not half of it), that running out is an error at the prompt after which everything works and
 * memory is back, and that the memory a deep recursion had for its frames can go to a vector (the slabs of
 * small blocks come back to malloc when it fails).
 *
 * Built and run by tests/c_abi.rs. The runtime here is the 64-bit host build, whose own state is larger than
 * the Pico's (15 KB against 22 KB with the lane on the board); the figures are relative to what `.mem()` says is in use.
 */
#define _DARWIN_C_SOURCE
#define _DEFAULT_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "klong.h"

/* ---- a model of newlib's malloc over a fixed arena ---------------------------------------- */

#include "klong_heap_arena.h"

#define KH_MALLOC mock_malloc
#define KH_FREE mock_free
#define KH_MEMALIGN mock_memalign
#define KH_POISON 1 /* freed blocks are filled with 0xDD, new ones with 0xAA: a use after free reads garbage */
#include "klong_heap.h"
#include "klong_heap_arena_impl.h"

struct board {
    int to_board[2];
    int from_board[2];
};

static void h_write(void *ctx, const uint8_t *bytes, size_t len) {
    struct board *b = ctx;
    while (len > 0) {
        ssize_t n = write(b->from_board[1], bytes, len);
        if (n <= 0) abort();
        bytes += n, len -= (size_t)n;
    }
}
static size_t h_read(void *ctx, uint8_t *buf, size_t cap) {
    ssize_t n = read(((struct board *)ctx)->to_board[0], buf, cap);
    return n > 0 ? (size_t)n : 0;
}
static uint64_t h_now_ms(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u;
}
static void h_wait(void *ctx, uint32_t max_ms) {
    (void)ctx, (void)max_ms;
    usleep(200);
}

static struct board B;
static int failures;

static const char *io(const char *line) {
    static char out[16384];
    size_t got = 0, len = strlen(line);
    if (len > 0 && write(B.to_board[1], line, len) != (ssize_t)len) abort();
    for (int quiet = 0; quiet < 3;) {
        if (klong_step() == 1) quiet = 0; else quiet++;
        ssize_t n = read(B.from_board[0], out + got, sizeof out - 1 - got);
        if (n > 0) got += (size_t)n, quiet = 0;
    }
    out[got] = 0;
    return out;
}

/* evaluate one line; the text the console printed for it (the echo and the prompt cut off) */
static const char *eval(const char *expr) {
    static char line[1024], res[16384];
    snprintf(line, sizeof line, "%s\n", expr);
    const char *raw = io(line);
    const char *p = strstr(raw, "\r\n");
    p = p ? p + 2 : raw;
    snprintf(res, sizeof res, "%s", p);
    char *q = strstr(res, "klong> ");
    if (q) *q = 0;
    size_t n = strlen(res);
    while (n && (res[n - 1] == '\n' || res[n - 1] == '\r')) res[--n] = 0;
    return res;
}

static void check(int ok, const char *what, const char *got) {
    printf("%s %s%s%s\n", ok ? "ok  " : "!!  ", what, got ? ": " : "", got ? got : "");
    if (!ok) failures++;
}

static int works(long n) {
    char e[64];
    snprintf(e, sizeof e, "#!%ld", n);
    const char *r = eval(e);
    return strstr(r, "memory error") == NULL && strtol(r, NULL, 10) == n;
}

static long mem(int field) {
    const char *r = eval(".mem()"); /* [used peak limit] */
    long v[3] = {0, 0, 0};
    sscanf(r, "[%ld %ld %ld]", &v[0], &v[1], &v[2]);
    return v[field];
}

static long largest(void) {
    long lo = 0, hi = (long)KH_TESTED_ARENA / 8 + 1;
    while (hi - lo > 1) {
        long mid = (lo + hi) / 2;
        if (works(mid)) lo = mid; else hi = mid;
    }
    return lo;
}

int main(void) {
    mock_init();
    if (pipe(B.to_board) || pipe(B.from_board)) return 2;
    fcntl(B.to_board[0], F_SETFL, O_NONBLOCK);
    fcntl(B.from_board[0], F_SETFL, O_NONBLOCK);

    klong_config cfg = {
        .abi_version = KLONG_ABI_VERSION,
        .size = sizeof cfg,
        .ctx = &B,
        .alloc = kh_alloc,
        .free = kh_free,
        .stack_bytes = 512u << 10,
        .write = h_write,
        .read = h_read,
        .now_ms = h_now_ms,
        .wait = h_wait,
        .name = "pico-heap",
        .flags = KLONG_FLAG_CRLF,
    };
    cfg.heap_overhead = 0; /* as the firmware: the region has no header, malloc's 8 are in the slack */
    cfg.heap_round = 8;
    kh_limits(KH_TESTED_ARENA, &cfg.heap_limit, &cfg.heap_reserve);
    if (klong_init(&cfg) != KLONG_OK) return puts("!! klong_init failed"), 2;
    io("");
    long limit = (long)cfg.heap_limit, reserve = (long)cfg.heap_reserve;
    printf("arena %zu bytes, heap limit %ld, reserve %ld\n", (size_t)KH_TESTED_ARENA, limit, reserve);

    /* the largest vector: one allocation of its final size */
    long used0 = mem(0);
    long n_max = largest();
    long expect = (limit - reserve - used0) / 8;
    char msg[160];
    snprintf(msg, sizeof msg, "!n_max = %ld (%ld bytes, %.0f%% of the arena); limit - reserve - in use = %ld elements", n_max, n_max * 8, 100.0 * (double)(n_max * 8) / (double)KH_TESTED_ARENA, expect);
    check(n_max >= expect - 64 && n_max <= expect, "the largest !n is what the limit leaves, allocated once", msg);
    check(n_max * 8 * 2 > (long)KH_TESTED_ARENA * 6 / 10, "a vector can use more than half of the arena (no doubling, no copy)", NULL);

    /* an allocation over the limit fails at the prompt; then everything works and the memory is back */
    check(!works(n_max + 200) && strstr(eval("#!99999999"), "memory error") != NULL, "over the limit: a memory error", NULL);
    long used1 = mem(0);
    snprintf(msg, sizeof msg, "in use %ld before, %ld after the failure", used0, used1);
    check(labs(used1 - used0) < 2000, "after the error the memory is back", msg);
    check(works(1000), "after the error a small vector works", NULL);
    check(strstr(eval("a::!1000;#a"), "1000") != NULL && works(n_max - 1250) && !works(n_max - 800), "a vector held: another fits in what is left, not more", NULL);
    check(strstr(eval("a::0"), "memory error") == NULL && works(n_max - 150), "dropped, the largest fits again (what the error left in use is a few hundred bytes)", NULL);
    snprintf(msg, sizeof msg, "a::!%ld;#a", n_max - 150);
    check(strstr(eval(msg), "memory error") == NULL && strstr(eval("a::0"), "memory error") == NULL, "the largest vector assigned to a name, then dropped", NULL);
    check(labs(mem(0) - used0) < 2000, "dropping it returns it", NULL);
    check(strstr(eval("g::{x+1};g(2)"), "3") != NULL, "functions work after all of it", NULL);

    /* after a vector as large as there is (malloc's break is then at the top of the arena) and its going, small blocks have
     * their room still: the region of small blocks cannot grow past the break, and takes what malloc has free below it */
    snprintf(msg, sizeof msg, "a::!%ld;a::0", n_max - 150);
    eval(msg);
    check(strstr(eval("l::{$x}'!600;#l"), "600") != NULL, "after the largest vector has gone, a few hundred small blocks (strings) fit again", NULL);
    check(strstr(eval("l::0"), "memory error") == NULL, "and go", NULL);

    /* the frames of a deep recursion are small blocks: they come back for a vector */
    check(strstr(eval("f::{:[x;1+f(x-1);0]};f(40)"), "40") != NULL, "a recursion 40 levels deep (a level costs the 64-bit host about 1 KB, the Pico about 0.3 KB)", NULL);
    uint32_t drains0 = kh_drains;
    printf("   (counted in use: %ld before, %ld after the recursion; malloc's: %zu)\n", used0, mem(0), mock_used_bytes);
    { size_t cnt; size_t big = mock_largest_free(&cnt); printf("   (after the recursion: %zu free chunks, the largest %zu bytes, %zu bytes in malloc's use, %u bytes cached)\n", cnt, big, mock_used_bytes, (unsigned)kh_cached); }
    long used_after = mem(0);
    long n_after = largest();
    long expect_after = (limit - reserve - used_after) / 8;
    snprintf(msg, sizeof msg, "%ld elements after (%ld before; the limit allows %ld with what is in use now, %ld bytes: the definitions of f and g); the cache was given back %u times (%u bytes in all)", n_after, n_max, expect_after, used_after, (unsigned)(kh_drains - drains0), (unsigned)kh_drained_bytes);
    check(n_after >= expect_after - 500 && used_after - used0 < 5000, "after the recursion the frames are malloc's again: the largest vector is what the limit allows, within the fragmentation of 12 free chunks", msg);
    check(kh_cached <= KH_CACHE_MAX, "the cache of free small blocks stays within its 8 KB (and is given back when malloc fails)", NULL);
    check(works(n_after), "and the vector is there", NULL);

    printf("-- %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
