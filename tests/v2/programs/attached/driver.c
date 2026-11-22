/* Host integration fixture for the Pico C ABI. The transport endpoint is the generated HLS
   kernel in C simulation. Count submissions independently of the result seen by Klong. */
#define _POSIX_C_SOURCE 200809L
#include "klong.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void probe_energy_kernel(const int64_t *in, int64_t *out, unsigned n, unsigned reset, unsigned *status);

typedef struct {
    const char *input;
    int submissions;
    int fail_completion;
    int compatible;
} Board;

static void *allocate(void *ctx, size_t size, size_t align) {
    (void)ctx;
    void *p = NULL;
    if (align < sizeof(void *)) align = sizeof(void *);
    return posix_memalign(&p, align, size ? size : 1) == 0 ? p : NULL;
}
static void release(void *ctx, void *p, size_t size, size_t align) {
    (void)ctx; (void)size; (void)align;
    free(p);
}
static size_t receive(void *ctx, uint8_t *buf, size_t cap) {
    Board *b = ctx;
    size_t n = strlen(b->input);
    if (n > cap) n = cap;
    memcpy(buf, b->input, n);
    b->input += n;
    return n;
}
static void transmit(void *ctx, const uint8_t *buf, size_t len) {
    (void)ctx;
    fwrite(buf, 1, len, stdout);
}
static uint64_t now(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}
static int32_t probe(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)a;
    if (n) return 1;
    out->tag = KLONG_T_INT;
    out->i = ((Board *)ctx)->compatible;
    return 0;
}
static int32_t submit(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    Board *b = ctx;
    if (n != 1 || a[0].tag != KLONG_T_INTS || a[0].rank != 1 || a[0].len != 4) {
        out->data = "expected four integers";
        return 1;
    }
    b->submissions++;
    unsigned status = 0;
    int64_t result;
    probe_energy_kernel(a[0].data, &result, 1, 1, &status);
    if (status || b->fail_completion) {
        out->data = "kernel completion unavailable";
        return 3;
    }
    out->tag = KLONG_T_INT;
    out->i = result;
    return 0;
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    Board b = {.input = argv[2], .fail_completion = strcmp(argv[1], "fault") == 0,
               .compatible = strcmp(argv[1], "attached") == 0 || strcmp(argv[1], "fault") == 0};
    klong_config config = {
        .abi_version = KLONG_ABI_VERSION, .size = sizeof config, .ctx = &b,
        /* Budgets from the linked RP2040 bench firmware. The host process supplies the memory. */
        .alloc = allocate, .free = release, .heap_limit = 96u << 10,
        .heap_reserve = 16u << 10, .stack_bytes = 108u << 10,
        .write = transmit, .read = receive, .now_ms = now,
        .name = "board-test", .flags = KLONG_FLAG_DUMB
    };
    if (klong_init(&config) != KLONG_OK) return 3;
    if (klong_extern_register("energy4_ready", probe, &b, 0) != KLONG_OK) return 4;
    if (strcmp(argv[1], "absent") && klong_extern_register("fpga_energy4", submit, &b, 0) != KLONG_OK) return 4;
    for (int step = 0; step < 1000; step++) {
        int active = klong_step();
        if (active < 0) return 5;
        if (!active && !*b.input) break;
    }
    fprintf(stderr, "submissions=%d\n", b.submissions);
    return *b.input ? 6 : 0;
}
