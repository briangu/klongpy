/*
 * Klong on a Raspberry Pi Pico (RP2040) of the 4-Pico bench: a node on a packet switch.
 *
 * The lane: uart0, GP16 TX and GP17 RX, 115200 8N1, nothing else on those pins and no other
 * use of uart0 (the SDK's stdio over UART is switched off in CMakeLists.txt). RUN and SWD are
 * not touched. The receive side is interrupt-driven into a 4 KB ring, so frames arriving
 * while the interpreter evaluates are not lost (the lane carries at most 2.9 KB/s).
 *
 * The runtime is libklong_ffi.a (Rust, native/klong-ffi, feature `lane`) behind klong.h. It
 * finds the 16-byte frames, learns the node number, answers PING and sends STATUS, carries
 * datagrams, and runs Klong's kdb+ IPC (`.cli("node:2")`, `.srv("node:")`) and a console
 * (datagram channel 1) over them. The console is also on USB CDC for a session with a cable.
 *
 * Memory: the interpreter runs on a 48 KB stack of its own (the SDK's default stack is 2 KB),
 * its heap is newlib's malloc, limited to KLONG_HEAP_LIMIT.
 */
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/irq.h"
#include "hardware/structs/rosc.h"
#include "hardware/uart.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

#include "klong.h"

/* The lane. GP16/GP17 are uart0 TX/RX on the RP2040. */
#define LANE_UART uart0
#define LANE_TX_GPIO 16
#define LANE_RX_GPIO 17
#define LANE_BAUD 115200
#define LANE_RING 4096u /* power of two: 1.4 s of a saturated lane */

#define KLONG_STACK_BYTES (48u * 1024)
#define KLONG_STACK_BUDGET (KLONG_STACK_BYTES - 6u * 1024) /* the rest: frames above klong_run, interrupts */
#define KLONG_HEAP_LIMIT (144u * 1024)
#define KLONG_HEAP_RESERVE (16u * 1024)

static uint8_t klong_stack[KLONG_STACK_BYTES] __attribute__((aligned(8)));

/* ---- the lane: UART interrupt -> ring -> klong_lane_read ---------------------------------- */

static uint8_t ring[LANE_RING];
static volatile uint32_t ring_head, ring_tail;
static volatile uint32_t ring_overruns, fifo_overruns, line_errors;

static void __not_in_flash_func(lane_irq)(void) {
    uart_hw_t *hw = uart_get_hw(LANE_UART);
    while (!(hw->fr & UART_UARTFR_RXFE_BITS)) {
        uint32_t dr = hw->dr; /* data in bits 0..7, this byte's error flags above */
        if (dr & UART_UARTDR_OE_BITS) fifo_overruns++;
        if (dr & (UART_UARTDR_FE_BITS | UART_UARTDR_BE_BITS | UART_UARTDR_PE_BITS)) {
            line_errors++;
            continue;
        }
        uint32_t head = ring_head;
        if (head - ring_tail == LANE_RING) {
            ring_overruns++;
            continue;
        }
        ring[head % LANE_RING] = (uint8_t)dr;
        ring_head = head + 1;
    }
    hw->icr = UART_UARTICR_RXIC_BITS | UART_UARTICR_RTIC_BITS | UART_UARTICR_OEIC_BITS;
}

static void lane_init(void) {
    uart_init(LANE_UART, LANE_BAUD);
    uart_set_format(LANE_UART, 8, 1, UART_PARITY_NONE);
    uart_set_hw_flow(LANE_UART, false, false);
    uart_set_fifo_enabled(LANE_UART, true);
    gpio_set_function(LANE_TX_GPIO, GPIO_FUNC_UART);
    gpio_set_function(LANE_RX_GPIO, GPIO_FUNC_UART);
    gpio_pull_up(LANE_RX_GPIO); /* idle high if the router is unplugged or held in reset */
    irq_set_exclusive_handler(UART0_IRQ, lane_irq);
    irq_set_enabled(UART0_IRQ, true);
    uart_set_irqs_enabled(LANE_UART, true, false); /* receive, and receive timeout; no transmit interrupt */
}

static size_t lane_read(void *ctx, uint8_t *buf, size_t cap) {
    (void)ctx;
    size_t n = 0;
    uint32_t tail = ring_tail;
    while (n < cap && tail != ring_head) {
        buf[n++] = ring[tail % LANE_RING];
        tail++;
    }
    ring_tail = tail;
    return n;
}

/* The transmitter's 32-byte FIFO takes what fits; the library offers the rest again. */
static size_t lane_write(void *ctx, const uint8_t *buf, size_t len) {
    (void)ctx;
    uart_hw_t *hw = uart_get_hw(LANE_UART);
    size_t n = 0;
    while (n < len && !(hw->fr & UART_UARTFR_TXFF_BITS)) hw->dr = buf[n++];
    return n;
}

/* ---- klong.h callbacks ------------------------------------------------------------------- */

static void *k_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    return align <= 8 ? malloc(size) : memalign(align, size);
}
static void k_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx, (void)size, (void)align;
    free(ptr);
}

/* The console on USB CDC (for a session with a cable): raw bytes, nothing is echoed or
 * translated by the SDK. With no host attached, output is dropped. */
static void k_write(void *ctx, const uint8_t *bytes, size_t len) {
    (void)ctx;
    if (!stdio_usb_connected()) return;
    for (size_t i = 0; i < len; i++) putchar_raw(bytes[i]);
    stdio_flush();
}
/* Ctrl-T on the USB console prints the lane's counters instead of reaching Klong. */
static void print_stats(void) {
    klong_lane_stats st;
    size_t used, peak, limit;
    klong_lane_stats_get(&st);
    klong_heap(&used, &peak, &limit);
    printf("\r\nlane: node %d  frames rx %lu tx %lu ignored %lu pings %lu  bad crc %lu, noise bytes skipped %lu\r\n", (int)klong_lane_node(), (unsigned long)st.frames_rx,
           (unsigned long)st.frames_tx, (unsigned long)st.frames_ignored, (unsigned long)st.pings, (unsigned long)st.bad_frames, (unsigned long)st.skipped_bytes);
    printf("datagrams: delivered %lu duplicate %lu broken %lu refused %lu resent %lu  ipc in %lu out %lu, inbox overflow %lu  console peer %d dropped %lu\r\n",
           (unsigned long)st.datagrams_delivered, (unsigned long)st.datagrams_duplicate, (unsigned long)st.datagrams_broken, (unsigned long)st.datagrams_refused,
           (unsigned long)st.datagrams_resent, (unsigned long)st.ipc_in, (unsigned long)st.ipc_out, (unsigned long)st.inbox_overflow, (int)st.console_peer, (unsigned long)st.console_dropped);
    printf("uart: ring overruns %lu, fifo overruns %lu, line errors %lu  heap used %lu peak %lu of %lu\r\n", (unsigned long)ring_overruns, (unsigned long)fifo_overruns,
           (unsigned long)line_errors, (unsigned long)used, (unsigned long)peak, (unsigned long)limit);
    stdio_flush();
}
static size_t k_read(void *ctx, uint8_t *buf, size_t cap) {
    (void)ctx;
    size_t n = 0;
    while (n < cap) {
        int c = getchar_timeout_us(0);
        if (c == PICO_ERROR_TIMEOUT) break;
        if (c == 0x14) {
            print_stats();
            continue;
        }
        buf[n++] = (uint8_t)c;
    }
    return n;
}
static uint64_t k_now_ms(void *ctx) {
    (void)ctx;
    return time_us_64() / 1000u;
}
static void k_wait(void *ctx, uint32_t max_ms) {
    (void)ctx;
    /* the lane is read from an interrupt, so a short sleep loses nothing; 1 ms keeps the latency of a frame low */
    sleep_us(max_ms == 0 ? 0 : (max_ms > 1 ? 1000 : max_ms * 1000u));
}
static void k_reset(void *ctx, int32_t code) {
    (void)ctx, (void)code;
    watchdog_reboot(0, 0, 10); /* the RP2040 restarts, as .x(code) means; the lane is silent for the boot */
    for (;;) tight_loop_contents();
}
/* Bits from the ring oscillator's random bit and the microsecond clock. */
static uint64_t random_bits(void) {
    uint64_t v = time_us_64();
    for (int i = 0; i < 64; i++) {
        v = (v << 1) ^ (rosc_hw->randombit & 1u);
        busy_wait_us(1 + (v & 3u));
    }
    return v ^ (v >> 29) * 0x9e3779b97f4a7c15ull;
}
static uint64_t k_entropy(void *ctx) {
    (void)ctx;
    return random_bits();
}
static void k_fatal(void *ctx, const char *msg, size_t len) {
    (void)ctx;
    if (stdio_usb_connected()) {
        printf("\r\nklong fatal: %.*s\r\n", (int)len, msg);
        stdio_flush();
    }
    watchdog_reboot(0, 0, 100);
    for (;;) tight_loop_contents();
}

static void klong_main(void) {
    klong_config cfg;
    klong_lane_config lane;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = k_alloc;
    cfg.free = k_free;
    cfg.heap_limit = KLONG_HEAP_LIMIT;
    cfg.heap_reserve = KLONG_HEAP_RESERVE;
    cfg.stack_bytes = KLONG_STACK_BUDGET;
    cfg.write = k_write;
    cfg.read = k_read;
    cfg.now_ms = k_now_ms;
    cfg.wait = k_wait;
    cfg.reset = k_reset;
    cfg.entropy = k_entropy;
    cfg.fatal = k_fatal;
    cfg.name = "pico-bench";
    cfg.flags = KLONG_FLAG_CRLF; /* files live in RAM (MemFs), in the heap: lost at reset */
    if (klong_init(&cfg) != KLONG_OK) k_fatal(NULL, "klong_init", 10);

    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = lane_read;
    lane.lane_write = lane_write;
    lane.node = -1; /* learned from the first frame the router sends to this port */
    lane.seed = (uint32_t)random_bits();
    lane.flags = 0;
    if (klong_lane_init(&lane) != KLONG_OK) k_fatal(NULL, "klong_lane_init", 15);
    klong_run();
    for (;;) tight_loop_contents();
}

/* Move the stack pointer to the 48 KB array and call klong_main there: the SDK's stack is 2 KB. */
static void __attribute__((noreturn, noinline)) on_big_stack(void (*fn)(void)) {
    uint32_t top = ((uint32_t)klong_stack + sizeof klong_stack) & ~7u;
    __asm volatile("msr msp, %0\n\tbx %1\n\t" ::"r"(top), "r"(fn) : "memory");
    for (;;) {
    }
}

int main(void) {
    /* USB CDC for the console; stdio over uart is off, so uart0 belongs to the lane alone */
    stdio_usb_init();
    lane_init();
    on_big_stack(klong_main);
}
