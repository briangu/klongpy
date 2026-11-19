/*
 * Host check of the C ABI (klong.h): the runtime linked into a plain C program, its console
 * on two pipes, driven as a terminal and an upload tool drive a board. Prints the
 * transcript (0x01 shown as ^A, CR dropped) and exits 0 when every exchange matched.
 * Built and run by tests/c_abi.rs (cargo test -p klong-ffi -- --nocapture shows the
 * transcript) or by hand, from native/:
 *
 *   cargo build --release -p klong-ffi
 *   cc -std=c11 -Wall -Wextra -Iklong-ffi/include klong-ffi/tests/host.c \
 *      target/release/libklong_ffi.a -o /tmp/klong-host && /tmp/klong-host
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

struct board {
    int to_board[2];   /* the tool writes [1], the board reads [0] */
    int from_board[2]; /* the board writes [1], the tool reads [0] */
    long allocs, frees, idles, waits, resets;
    uint64_t fake_ms;  /* added to the clock: lets the test skip ahead */
};

static void *h_alloc(void *ctx, size_t size, size_t align) {
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (posix_memalign(&p, align, size ? size : 1) != 0) return NULL;
    ((struct board *)ctx)->allocs++;
    return p;
}
static void h_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)size, (void)align;
    ((struct board *)ctx)->frees++;
    free(ptr);
}
static void h_write(void *ctx, const uint8_t *bytes, size_t len) {
    struct board *b = ctx;
    while (len > 0) {
        ssize_t n = write(b->from_board[1], bytes, len);
        if (n <= 0) abort();
        bytes += n, len -= (size_t)n;
    }
}
static size_t h_read(void *ctx, uint8_t *buf, size_t cap) {
    ssize_t n = read(((struct board *)ctx)->to_board[0], buf, cap); /* O_NONBLOCK */
    return n > 0 ? (size_t)n : 0;
}
static uint64_t h_now_ms(void *ctx) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000u + (uint64_t)t.tv_nsec / 1000000u + ((struct board *)ctx)->fake_ms;
}
static void h_idle(void *ctx) { ((struct board *)ctx)->idles++; }
static void h_wait(void *ctx, uint32_t max_ms) {
    (void)max_ms;
    ((struct board *)ctx)->waits++;
    usleep(1000);
}
static void h_reset(void *ctx, int32_t code) {
    (void)code;
    ((struct board *)ctx)->resets++; /* returns: the program only stops */
}

static uint32_t crc32(const uint8_t *p, size_t n) {
    uint32_t c = 0xffffffffu;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static int failures;

/* Send `len` bytes, turn the main loop until it has nothing to do, return what the board
 * sent (NUL-terminated, static buffer). */
static const char *io(struct board *b, const void *bytes, size_t len) {
    static char out[65536];
    size_t got = 0;
    if (len > 0 && write(b->to_board[1], bytes, len) != (ssize_t)len) abort();
    for (int quiet = 0; quiet < 3;) {
        if (klong_step() == 1) quiet = 0; else quiet++;
        ssize_t n = read(b->from_board[0], out + got, sizeof out - 1 - got);
        if (n > 0) got += (size_t)n, quiet = 0;
    }
    out[got] = 0;
    return out;
}

static void show(const char *who, const char *s) {
    printf("%s ", who);
    for (; *s; s++) {
        if (*s == 1) fputs("^A", stdout);
        else if (*s == '\r') continue;
        else if (*s == '\n' && s[1]) printf("\n%s ", who);
        else putchar(*s);
    }
    if (s[-1] != '\n') putchar('\n');
}

/* One exchange: send, and compare the answer: all of it, or (`part` set) `want` must be
 * somewhere in it. */
static const char *expect(struct board *b, const char *send, size_t len, const char *want, int part) {
    if (len > 0) show(">>", send);
    const char *got = io(b, send, len);
    show("<<", got);
    if (part ? strstr(got, want) == NULL : strcmp(got, want) != 0) {
        printf("!! expected%s: ", part ? " somewhere" : "");
        show("", want);
        failures++;
    }
    return got;
}
#define EXPECT(b, send, want) expect(b, send, strlen(send), want, 0)
#define EXPECT_PART(b, send, want) expect(b, send, strlen(send), want, 1)

int main(void) {
    static struct board b;
    if (pipe(b.to_board) || pipe(b.from_board)) return 2;
    fcntl(b.to_board[0], F_SETFL, O_NONBLOCK);
    fcntl(b.from_board[0], F_SETFL, O_NONBLOCK);

    klong_config cfg = {
        .abi_version = KLONG_ABI_VERSION,
        .size = sizeof cfg,
        .ctx = &b,
        .alloc = h_alloc,
        .free = h_free,
        .heap_limit = 4u << 20,
        .heap_reserve = 16u << 10,
        .stack_bytes = 256u << 10,
        .write = h_write,
        .read = h_read,
        .now_ms = h_now_ms,
        .idle = h_idle,
        .wait = h_wait,
        .reset = h_reset,
        .name = "host-c",
        .flags = KLONG_FLAG_CRLF,
    };

    /* the interface refuses misuse */
    klong_config bad = cfg;
    bad.abi_version = 99;
    if (klong_step() != -1 || klong_init(NULL) != KLONG_ERR_ARG || klong_init(&bad) != KLONG_ERR_ARG) {
        puts("!! misuse before init not refused");
        failures++;
    }
    bad = cfg, bad.read = NULL;
    if (klong_init(&bad) != KLONG_ERR_ARG) failures++, puts("!! missing callback accepted");
    if (klong_init(&cfg) != KLONG_OK) return puts("!! klong_init failed"), 2;
    if (klong_init(&cfg) != KLONG_ERR_STATE) failures++, puts("!! second init accepted");

    /* banner and prompt, as klong-sim prints them (with this board's name) */
    expect(&b, "", 0, "Klong 2.0.0-rc.1 (host-c, int64 float64)\r\nklong> ", 0);

    /* a terminal: echo, value, prompt */
    EXPECT(&b, "1+2\n", "1+2\r\n3\r\nklong> ");
    EXPECT(&b, "+/!100\n", "+/!100\r\n4950\r\nklong> ");
    EXPECT(&b, "2.5*4\n", "2.5*4\r\n10.0\r\nklong> ");

    /* upload with ^A put, then load and call it; the library is in the image */
    const char *prog = "sq::{x*x}\n.p(\"loaded\")\n";
    char head[96];
    snprintf(head, sizeof head, "\001put hello.kg %zu %08x\n", strlen(prog), (unsigned)crc32((const uint8_t *)prog, strlen(prog)));
    EXPECT_PART(&b, head, "\r\n\001go\r\n"); /* the command line is echoed first */
    char want[96];
    snprintf(want, sizeof want, "\001ok %zu %08x\r\nklong> ", strlen(prog), (unsigned)crc32((const uint8_t *)prog, strlen(prog)));
    EXPECT(&b, prog, want);
    EXPECT(&b, ".l(\"hello\");sq(12)\n", ".l(\"hello\");sq(12)\r\nloaded\r\n144\r\nklong> ");
    EXPECT(&b, ".l(\"nstat\");nstat.mu([1 2 3 4])\n", ".l(\"nstat\");nstat.mu([1 2 3 4])\r\n2.5\r\nklong> ");

    /* a program sent with ^A run */
    const char *run = "sq(7)+1";
    snprintf(head, sizeof head, "\001run %zu %08x\n", strlen(run), (unsigned)crc32((const uint8_t *)run, strlen(run)));
    EXPECT_PART(&b, head, "\r\n\001go\r\n");
    EXPECT(&b, run, "50\r\n\001ok value 2\r\nklong> ");

    /* ver */
    const char *ver = EXPECT_PART(&b, "\001ver\n", "\001ok ver=2.0.0-rc.1 int=64 float=64 features=files,lib");
    /* (in a build of the whole workspace cargo adds the features other crates ask for) */
    if (!strstr(ver, "timers") || !strstr(ver, "platform=host-c") || !strstr(ver, " fs=1 ") || !strstr(ver, "heap_limit=4194304")) {
        puts("!! ver lacks timers, platform, fs or heap_limit");
        failures++;
    }

    /* limits: the stack budget and the heap limit are errors at the prompt */
    const char *r = EXPECT_PART(&b, "f::{f(x+1)};f(0)\n", "f::{f(x+1)};f(0)\r\n");
    if (!strstr(r, "range error")) failures++, puts("!! no range error");
    r = EXPECT_PART(&b, "!100000000\n", "!100000000\r\n");
    if (!strstr(r, "memory error")) failures++, puts("!! no memory error");

    /* timers run from klong_step; the wait callback is what klong_run sleeps in */
    EXPECT_PART(&b, "tk::0;cb::{tk::tk+1;tk<3};th::.timer(\"t\";0.01;cb)\n", "\r\nklong> ");
    b.fake_ms += 1000;
    for (int i = 0; i < 10; i++) klong_step(), b.fake_ms += 20;
    EXPECT(&b, "tk\n", "tk\r\n3\r\nklong> ");

    /* .x calls reset; this one returns, so the prompt comes back */
    EXPECT_PART(&b, ".x(0)\n", ".x(0)\r\n");
    EXPECT(&b, "sq(3)\n", "sq(3)\r\n9\r\nklong> ");

    size_t used, peak, limit;
    klong_heap(&used, &peak, &limit);
    printf("-- heap: used %zu, peak %zu, limit %zu; alloc %ld, free %ld; idle %ld, reset %ld\n", used, peak, limit, b.allocs, b.frees, b.idles, b.resets);
    if (used == 0 || peak < used || peak > limit || limit != (4u << 20) || b.idles == 0 || b.resets != 1) {
        puts("!! heap figures or callback counts are off");
        failures++;
    }
    printf("-- %s\n", failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
