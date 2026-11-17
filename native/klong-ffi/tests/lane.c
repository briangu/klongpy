/*
 * Host check of the lane part of the C ABI (klong.h, feature `lane`): the runtime linked into
 * a plain C program whose "UART" is two buffers, driven as the bench's supervisor drives a
 * Pico: a PING that numbers the node and is answered with a PONG, a STATUS a second later,
 * a console line sent in datagram frames on channel 1 and answered the same way, unknown
 * frames ignored, line noise between frames. Exits 0 when every exchange matched.
 *
 *   cargo test -p klong-ffi --features lane   (tests/c_abi.rs builds and runs it)
 */
#define _DARWIN_C_SOURCE
#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "klong.h"

static uint8_t rx[4096];        /* bytes the supervisor sends to the board */
static size_t rx_len, rx_pos;
static uint8_t tx[65536];       /* bytes the board sends to the supervisor */
static size_t tx_len;
static uint64_t now;
static int failures;

static void *h_alloc(void *ctx, size_t size, size_t align) {
    void *p = NULL;
    (void)ctx;
    if (align < sizeof(void *)) align = sizeof(void *);
    return posix_memalign(&p, align, size ? size : 1) == 0 ? p : NULL;
}
static void h_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx, (void)size, (void)align;
    free(ptr);
}
static void h_write(void *ctx, const uint8_t *b, size_t n) { (void)ctx, (void)b, (void)n; }
static size_t h_read(void *ctx, uint8_t *b, size_t cap) { (void)ctx, (void)b, (void)cap; return 0; }
static uint64_t h_now(void *ctx) { (void)ctx; return now; }
static size_t l_read(void *ctx, uint8_t *b, size_t cap) {
    size_t n = rx_len - rx_pos < cap ? rx_len - rx_pos : cap;
    (void)ctx;
    memcpy(b, rx + rx_pos, n);
    rx_pos += n;
    return n;
}
static size_t l_write(void *ctx, const uint8_t *b, size_t n) {
    (void)ctx;
    if (tx_len + n > sizeof tx) return 0;
    memcpy(tx + tx_len, b, n);
    tx_len += n;
    return n;
}

static uint16_t crc16(const uint8_t *p, size_t n) {
    uint16_t c = 0xffff;
    while (n--) {
        c ^= (uint16_t)(*p++) << 8;
        for (int k = 0; k < 8; k++) c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}
static uint32_t crc32(const uint8_t *p, size_t n) {
    uint32_t c = 0xffffffffu;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static void put_frame(uint8_t type, uint8_t src, uint8_t dst, uint16_t seq, uint8_t sub, const uint8_t *data, size_t len) {
    uint8_t f[16] = {0xa5, 0x5a, 1, type, src, dst, (uint8_t)seq, (uint8_t)(seq >> 8), sub, (uint8_t)len};
    uint16_t c;
    memcpy(f + 10, data, len);
    c = crc16(f + 2, 12);
    f[14] = (uint8_t)c;
    f[15] = (uint8_t)(c >> 8);
    memcpy(rx + rx_len, f, 16);
    rx_len += 16;
}

/* a message from the supervisor to node `dst` on `chan`, as START, DATA, END */
static void put_message(uint8_t dst, uint8_t chan, uint16_t id, const char *text) {
    size_t n = strlen(text), i;
    uint8_t start[4] = {(uint8_t)n, (uint8_t)(n >> 8), (uint8_t)(chan == 1 ? 1 : 0), chan}; /* console input asks for an ACK */
    uint32_t crc = crc32((const uint8_t *)text, n);
    uint8_t end[4] = {(uint8_t)crc, (uint8_t)(crc >> 8), (uint8_t)(crc >> 16), (uint8_t)(crc >> 24)};
    put_frame(9, 16, dst, id, 0xd0, start, 4);
    for (i = 0; i * 4 < n; i++) put_frame(9, 16, dst, (uint16_t)i, 0xd1, (const uint8_t *)text + i * 4, n - i * 4 < 4 ? n - i * 4 : 4);
    put_frame(9, 16, dst, id, 0xd2, end, 4);
}

static void run(int turns, uint64_t ms_per_turn) {
    while (turns--) {
        now += ms_per_turn;
        klong_step();
    }
}

/* every frame the board sent since `from`, reassembled: datagrams to the supervisor on `chan` */
static size_t collect(size_t from, uint8_t chan, char *out, size_t cap, int *pongs, int *statuses) {
    size_t i, len = 0;
    int cur = -1;
    for (i = from; i + 16 <= tx_len; i += 16) {
        const uint8_t *f = tx + i;
        if (f[0] != 0xa5 || f[1] != 0x5a || crc16(f + 2, 12) != (uint16_t)(f[14] | f[15] << 8)) {
            printf("bad frame from the board at %zu\n", i);
            failures++;
            continue;
        }
        if (f[3] == 2 && pongs) (*pongs)++;
        if (f[3] == 3 && statuses) (*statuses)++;
        if (f[3] == 9 && f[8] == 0xd0 && f[5] == 16) cur = f[13];
        if (f[3] == 9 && f[8] == 0xd1 && f[5] == 16 && cur == chan && len + f[9] < cap) {
            memcpy(out + len, f + 10, f[9]);
            len += f[9];
        }
    }
    out[len] = 0;
    return len;
}

#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

/* The boot self-test on node 0: serving, STATUS flags 0x31, a report datagram to 16 on channel 1. */
static int selftest_main(void) {
    klong_config cfg;
    klong_lane_config lane;
    klong_lane_stats st;
    char text[512];
    int statuses = 0, flags_ok = 0;
    size_t i;

    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = h_alloc;
    cfg.free = h_free;
    cfg.heap_limit = 4u << 20;
    cfg.heap_reserve = 16384;
    cfg.stack_bytes = 1u << 20;
    cfg.write = h_write;
    cfg.read = h_read;
    cfg.now_ms = h_now;
    cfg.name = "selftest";
    cfg.flags = KLONG_FLAG_NO_FS;
    CHECK(klong_init(&cfg) == KLONG_OK, "init");
    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = l_read;
    lane.lane_write = l_write;
    lane.node = 0;
    lane.seed = 7;
    lane.baud = 4500000u;
    lane.flags = KLONG_LANE_SELFTEST;
    CHECK(klong_lane_init(&lane) == KLONG_OK, "lane init");
    run(40, 100);
    collect(0, 3, text, sizeof text, NULL, &statuses);
    for (i = 0; i + 16 <= tx_len; i += 16)
        if (tx[i + 3] == 3 && tx[i + 5] == 16 && tx[i + 9] == 4 && tx[i + 11] == 0x31) flags_ok++;
    CHECK(statuses >= 3 && flags_ok == statuses, "STATUS frames with flags 0x31: %d of %d", flags_ok, statuses);
    CHECK(strstr(text, "klong selftest serving node=0 calls=0") != NULL, "report: [%s]", text);
    printf("report: [%s]\n", text);
    klong_lane_stats_get(&st);
    CHECK(st.selftest == 0, "no verdict for a server");
    printf(failures ? "-- %d failures\n" : "-- ok\n", failures);
    return failures ? 1 : 0;
}

int main(int argc, char **argv) {
    klong_config cfg;
    klong_lane_config lane;
    klong_lane_stats st;
    char text[512], text2[128];
    int pongs = 0, statuses = 0;
    size_t mark;

    if (argc > 1 && strcmp(argv[1], "selftest") == 0) return selftest_main();
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = h_alloc;
    cfg.free = h_free;
    cfg.heap_limit = 4u << 20;
    cfg.heap_reserve = 16384;
    cfg.stack_bytes = 1u << 20;
    cfg.write = h_write;
    cfg.read = h_read;
    cfg.now_ms = h_now;
    cfg.name = "lane-test";
    cfg.flags = KLONG_FLAG_NO_FS;
    CHECK(klong_init(&cfg) == KLONG_OK, "init");

    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = l_read;
    lane.lane_write = l_write;
    lane.node = -1;
    lane.seed = 0x1234abcd;
    CHECK(klong_lane_init(&lane) == KLONG_OK, "lane init");
    CHECK(klong_lane_init(&lane) == KLONG_ERR_STATE, "second lane init");
    CHECK(klong_lane_node() == -1, "unnumbered at the start");
    run(5, 1);
    CHECK(tx_len == 0, "silent until numbered: %zu bytes", tx_len);

    /* noise, a broadcast PING (does not name a port), then a PING to port 2 */
    memcpy(rx, "\xa5\x5a\x01\x01garbage\xff\xa5", 13);
    rx_len = 13;
    put_frame(1, 16, 254, 1, 16, NULL, 0);
    put_frame(1, 16, 2, 7, 16, NULL, 0);
    run(5, 1);
    CHECK(klong_lane_node() == 2, "numbered 2, got %d", klong_lane_node());
    pongs = 0;
    collect(0, 0, text, sizeof text, &pongs, &statuses);
    CHECK(pongs == 1, "one PONG, got %d", pongs);
    CHECK(tx_len == 16 && tx[3] == 2 && tx[4] == 2 && tx[5] == 16 && tx[6] == 7, "PONG fields");

    /* a second later: STATUS to the supervisor */
    mark = tx_len;
    run(12, 100);
    statuses = 0;
    collect(mark, 0, text, sizeof text, NULL, &statuses);
    CHECK(statuses >= 1, "STATUS after a second, got %d", statuses);

    /* frames the node does not know are ignored, a console line is answered on channel 1 */
    mark = tx_len;
    rx_len = rx_pos = 0;
    put_frame(9, 16, 2, 9, 0xa0, (const uint8_t *)"\x01\x02\x03\x04", 4);
    put_frame(9, 16, 2, 10, 0xe1, (const uint8_t *)"\x01\x00\x00\x00", 4);
    put_frame(3, 16, 2, 11, 16, (const uint8_t *)"\x00\x01", 2);
    put_message(2, 1, 100, "1+2\r");
    run(40, 5);
    collect(mark, 1, text, sizeof text, NULL, NULL);
    CHECK(strstr(text, "3") != NULL, "console reply on the lane: [%s]", text);
    CHECK(strstr(text, "klong> ") == NULL && strstr(text, "1+2") == NULL, "no prompt and no echo for a datagram peer: [%s]", text);
    collect(mark, 2, text2, sizeof text2, NULL, NULL);
    CHECK(strcmp(text2, "done 100 ok") == 0, "done on channel 2: [%s]", text2);
    printf("console reply: [%s]\n", text);

    /* a second line, a longer reply */
    mark = tx_len;
    rx_len = rx_pos = 0;
    put_message(2, 1, 101, "+/!1000\r");
    run(80, 5);
    collect(mark, 1, text, sizeof text, NULL, NULL);
    CHECK(strstr(text, "499500") != NULL, "second reply: [%s]", text);
    collect(mark, 2, text2, sizeof text2, NULL, NULL);
    CHECK(strcmp(text2, "done 101 ok") == 0, "done on channel 2: [%s]", text2);
    /* an error, hello and int when idle */
    mark = tx_len;
    rx_len = rx_pos = 0;
    put_message(2, 1, 102, "1+\"a\"\r");
    put_message(2, 2, 103, "hello");
    put_message(2, 2, 104, "int");
    run(80, 5);
    collect(mark, 2, text2, sizeof text2, NULL, NULL);
    CHECK(strstr(text2, "done 102 error") != NULL && strstr(text2, "klong 0.0.1 node=2 lane-test int64 float64") != NULL, "error, hello: [%s]", text2);
    collect(mark, 1, text, sizeof text, NULL, NULL);
    CHECK(strstr(text, "error") != NULL, "the error text is on channel 1: [%s]", text);

    klong_lane_stats_get(&st);
    printf("lane: rx %u tx %u ignored %u pings %u delivered %u console peer %d bad %u skipped %u\n", st.frames_rx, st.frames_tx, st.frames_ignored, st.pings, st.datagrams_delivered, st.console_peer, st.bad_frames, st.skipped_bytes);
    CHECK(st.console_peer == 16, "console peer");
    CHECK(st.datagrams_delivered == 5, "five datagrams delivered, got %u", st.datagrams_delivered);
    CHECK(st.frames_ignored >= 1, "ignored frames");
    CHECK(st.skipped_bytes > 0, "noise skipped");

    printf(failures ? "-- %d failures\n" : "-- ok\n", failures);
    return failures ? 1 : 0;
}
