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
 * Memory (the whole of the 264 KB of RAM; the figures are in README.md): the code that runs most (the evaluator)
 * is copied to RAM at start, 17 KB of .data; the lane's rings are 24 KB; the interpreter runs on a stack of
 * KLONG_STACK_BYTES that starts at the top of RAM, in the 8 KB of the two scratch banks, and grows down; the heap
 * is everything between the end of .bss and the bottom of that stack: the first KH_SDK_BYTES are newlib's (the SDK's own
 * malloc users, through the _sbrk below), the rest is the runtime's pool (klong_heap.h: its own allocator, one address-ordered
 * list with coalescing, no malloc), and the runtime's heap limit is made from its size.
 */
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/irq.h"
#include "hardware/structs/rosc.h"
#include "hardware/sync.h"
#include "hardware/uart.h"
#include "hardware/structs/watchdog.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "pico/stdio_usb.h"

#include "klong.h"
#ifndef KLONG_C_EXAMPLES
#define KLONG_C_EXAMPLES 1 /* the compiled static-profile examples as `.c.*`: see klong_c.h */
#endif
#if KLONG_C_EXAMPLES
#include "klong_c.h"
#endif

/* The runtime's memory is not newlib's: it has a pool of its own (klong_heap.h). What the SDK itself takes from malloc (stdio's
 * buffer when a console is on USB, a few small things) comes from the first KH_SDK_BYTES of the heap through `_sbrk` below; if
 * that is not enough the SDK's pico_malloc panics, as it always has. */
#ifndef KLONG_HEAP_DIAG
#define KLONG_HEAP_DIAG 1 /* `.heap`, and the allocator's table of live blocks: see klong_heap.h and docs/native/pico-bench.md */
#endif
#if KLONG_HEAP_DIAG
#define KH_DIAG 1
#endif
#include "klong_heap.h"
#include "klong_memory.h"

/* The lane. GP16/GP17 are uart0 TX/RX on the RP2040. */
#define LANE_UART uart0
#define LANE_TX_GPIO 16
#define LANE_RX_GPIO 17
#ifndef LANE_BAUD
#define LANE_BAUD 4500000u /* 27 MHz / 6 at the router; uart_init(uart0, 4500000) gives 4.505 Mbaud at 125 MHz clk_peri */
#endif
#ifndef KLONG_SELFTEST
#define KLONG_SELFTEST 1 /* the boot self-test: see README.md */
#endif

/* Receive: at 4.5 Mbaud a byte arrives every 2.2 us, a 16-byte frame every 36 us, and the 32-byte
 * hardware FIFO fills in 71 us. The interrupt (highest priority, FIFO level 8 bytes, plus the
 * receive-timeout interrupt for the tail of a burst) moves bytes into a 16 KB ring: 1024 frames,
 * 37 ms of a lane that never pauses. The main loop is away that long only inside one primitive
 * of the interpreter that does not call `idle` (a sort or a sum over a huge vector, say);
 * the runtime reads the ring at every turn of its loop, every 4096 steps of an evaluation and
 * inside every IPC call. What does not fit is dropped and counted; the layers above recover by
 * retransmission. Bytes are not parsed in the interrupt: that would put the CRC and the datagram
 * layer (Rust, with its state) at interrupt level for a saving that a ring of this size makes
 * unnecessary, and the ring costs 16 KB of the 264. */
#define LANE_RX_RING 16384u
/* Transmit: a ring of 8 KB drained by the transmit interrupt (FIFO half empty), so that the
 * interpreter never waits for the wire: klong_lane_write copies and returns. The library lets
 * at most 12 ms of wire time wait in it (lane_tx_limit): a deeper queue only delays what has to
 * go out behind it, an ACK behind a full window, and a shallower one lets the lane idle while
 * the main loop is away. */
#define LANE_TX_RING 8192u

/* The stack: KLONG_STACK_BYTES from the top of RAM (__StackTop, the end of the second scratch bank) down. About 870 bytes per
 * level of Klong recursion (the frames on the path, read from the disassembly: 864; measured on the boards with the limit
 * the lane gave: 90 levels worked and 95 did not, at 80,640 bytes), so 110 levels of a simple recursion are about 97 KB.
 * KLONG_STACK_KEEP of what is given to the evaluator is kept for what runs below its deepest level (a walk over nested
 * data, the lane's pump: about 3 KB; the data nesting allowed is made from it), and KLONG_STACK_ABOVE is the rest, for the
 * frames above klong_run and the interrupts, which run on this stack. */
#define KLONG_STACK_ABOVE (4u * 1024)
#define KLONG_STACK_BUDGET (KLONG_STACK_BYTES - KLONG_STACK_ABOVE)
#define KLONG_STACK_KEEP (16u * 1024)

extern char end;         /* the first byte after .data and .bss: the heap starts here */
extern char __StackTop;  /* the top of RAM */
#define STACK_BOTTOM ((char *)((uintptr_t)&__StackTop - KLONG_STACK_BYTES))

/* The heap: from `end` (rounded to 8) the SDK's KH_SDK_BYTES for newlib's malloc, then the runtime's pool to the bottom of the
 * interpreter's stack, which starts in the SDK's scratch banks (they are the SDK's own stacks, unused after boot). */
#define SDK_START ((char *)(((uintptr_t)&end + 7u) & ~(uintptr_t)7u))
#define SDK_STOP (SDK_START + KH_SDK_BYTES)
static char *sdk_brk;
void *_sbrk(int incr) {
    if (sdk_brk == NULL) sdk_brk = SDK_START;
    char *prev = sdk_brk;
    if (incr > 0 && (size_t)(SDK_STOP - prev) < (size_t)incr) return (void *)-1;
    if (incr < 0 && (size_t)(prev - SDK_START) < (size_t)-incr) return (void *)-1;
    sdk_brk += incr;
    return prev;
}

/* ---- the safety net: a fault, a overrun stack and a hang are seen, and the node restarts and says why ------------- */

/* Why the chip restarted is kept in the watchdog's scratch registers 0 to 3 (they survive a watchdog reset, not a power-on;
 * 4 to 7 are the SDK's): [0] a magic and the kind, [1] and [2] the program counter and link register of a fault, or the
 * first 8 characters of a fatal message. The next boot turns it into text for the self-test's reports (` boot=...`),
 * which the supervisor reads on channel 3: nothing here needs a cable. */
#define NOTE_MAGIC 0x4b4c4e00u /* "KLN" and the kind in the low byte */
enum { NOTE_HARDFAULT = 1, NOTE_STACK = 2, NOTE_FATAL = 3 };
static char boot_note[64];

static void note_set(uint32_t kind, uint32_t a, uint32_t b) {
    watchdog_hw->scratch[0] = NOTE_MAGIC | kind;
    watchdog_hw->scratch[1] = a;
    watchdog_hw->scratch[2] = b;
}

/* After the HardFault: where it happened. r0 is the stacked frame: r0-r3, r12, lr, pc, psr. */
void __attribute__((used)) fault_c(uint32_t *frame) {
    note_set(NOTE_HARDFAULT, frame[6], frame[5]);
    watchdog_reboot(0, 0, 1);
    for (;;) {
    }
}
void __attribute__((naked)) isr_hardfault(void) {
    __asm volatile("mov r0, lr\n\t"
                   "movs r1, #4\n\t"
                   "tst r0, r1\n\t"
                   "beq 1f\n\t"
                   "mrs r0, psp\n\t"
                   "b 2f\n\t"
                   "1: mrs r0, msp\n\t"
                   "2: ldr r1, =fault_c\n\t"
                   "bx r1\n\t");
}

/* Words at the bottom of the stack, which the heap's end (STACK_BOTTOM) stops short of: a recursion that got past the
 * stack budget, or anything that writes below it, changes them, and the idle path sees it within a turn. */
#define CANARY_WORDS 8u
#define CANARY 0xc0de5afeu
static void canary_set(void) {
    volatile uint32_t *w = (volatile uint32_t *)(uintptr_t)STACK_BOTTOM;
    for (uint32_t i = 0; i < CANARY_WORDS; i++) w[i] = CANARY ^ i;
}
static int canary_ok(void) {
    volatile uint32_t *w = (volatile uint32_t *)(uintptr_t)STACK_BOTTOM;
    for (uint32_t i = 0; i < CANARY_WORDS; i++)
        if (w[i] != (CANARY ^ i)) return 0;
    return 1;
}

/* The text of the reason of this boot, from the scratch registers or the watchdog's flags; empty after a power-on. */
static void boot_note_read(void) {
    uint32_t m = watchdog_hw->scratch[0];
    boot_note[0] = 0;
    if ((m & 0xffffff00u) == NOTE_MAGIC) {
        uint32_t a = watchdog_hw->scratch[1], b = watchdog_hw->scratch[2];
        switch (m & 0xff) {
        case NOTE_HARDFAULT: snprintf(boot_note, sizeof boot_note, "hardfault pc=%08lx lr=%08lx", (unsigned long)a, (unsigned long)b); break;
        case NOTE_STACK: snprintf(boot_note, sizeof boot_note, "stack"); break;
        default: snprintf(boot_note, sizeof boot_note, "fatal %.4s%.4s", (const char *)&a, (const char *)&b); break;
        }
        watchdog_hw->scratch[0] = 0;
    } else if (watchdog_enable_caused_reboot()) {
        snprintf(boot_note, sizeof boot_note, "watchdog");
    } else if (watchdog_caused_reboot()) {
        snprintf(boot_note, sizeof boot_note, "reset");
    }
}

/* ---- the lane: UART interrupts <-> rings ------------------------------------------------- */

static uint8_t rx_ring[LANE_RX_RING];
static uint8_t tx_ring[LANE_TX_RING];
static volatile uint32_t rx_head, rx_tail, tx_head, tx_tail;
/* what klong_lane_stats reports as the driver's counters: rx ring overruns, rx FIFO overruns, line errors, tx ring full */
static volatile uint32_t lane_counters[4];
static uint32_t lane_actual_baud, lane_tx_limit;

static void __not_in_flash_func(lane_irq)(void) {
    uart_hw_t *hw = uart_get_hw(LANE_UART);
    while (!(hw->fr & UART_UARTFR_RXFE_BITS)) {
        uint32_t dr = hw->dr; /* data in bits 0..7, this byte's error flags above */
        if (dr & UART_UARTDR_OE_BITS) lane_counters[1]++;
        if (dr & (UART_UARTDR_FE_BITS | UART_UARTDR_BE_BITS | UART_UARTDR_PE_BITS)) {
            lane_counters[2]++;
            continue;
        }
        uint32_t head = rx_head;
        if (head - rx_tail == LANE_RX_RING) {
            lane_counters[0]++;
            continue;
        }
        rx_ring[head % LANE_RX_RING] = (uint8_t)dr;
        rx_head = head + 1;
    }
    /* transmit: fill the FIFO from the ring; when the ring is empty stop the interrupt, which would
     * otherwise stay asserted while the FIFO is empty (klong_lane_write turns it on again) */
    uint32_t tail = tx_tail;
    while (tail != tx_head && !(hw->fr & UART_UARTFR_TXFF_BITS)) hw->dr = tx_ring[tail++ % LANE_TX_RING];
    tx_tail = tail;
    if (tail == tx_head) hw->imsc &= ~UART_UARTIMSC_TXIM_BITS;
    hw->icr = UART_UARTICR_RXIC_BITS | UART_UARTICR_RTIC_BITS | UART_UARTICR_OEIC_BITS;
}

static void lane_init(void) {
    lane_actual_baud = uart_init(LANE_UART, LANE_BAUD);
    uart_set_format(LANE_UART, 8, 1, UART_PARITY_NONE);
    uart_set_hw_flow(LANE_UART, false, false);
    uart_set_fifo_enabled(LANE_UART, true);
    gpio_set_function(LANE_TX_GPIO, GPIO_FUNC_UART);
    gpio_set_function(LANE_RX_GPIO, GPIO_FUNC_UART);
    gpio_pull_up(LANE_RX_GPIO); /* idle high if the router is unplugged or held in reset */
    /* FIFO levels: receive interrupt at 8 bytes (1/4), transmit interrupt at 16 or fewer left (1/2) */
    uart_get_hw(LANE_UART)->ifls = (1u << UART_UARTIFLS_RXIFLSEL_LSB) | (2u << UART_UARTIFLS_TXIFLSEL_LSB);
    irq_set_exclusive_handler(UART0_IRQ, lane_irq);
    irq_set_priority(UART0_IRQ, 0); /* above USB and everything else */
    irq_set_enabled(UART0_IRQ, true);
    uart_set_irqs_enabled(LANE_UART, true, false); /* receive and receive timeout; transmit is switched on by klong_lane_write */
    uint32_t limit = (uint32_t)((uint64_t)LANE_BAUD / 10u * 12u / 1000u) & ~15u; /* 12 ms of wire time, whole frames */
    lane_tx_limit = limit < 64u ? 64u : (limit > LANE_TX_RING ? LANE_TX_RING : limit);
}

static size_t lane_read(void *ctx, uint8_t *buf, size_t cap) {
    (void)ctx;
    size_t n = 0;
    uint32_t tail = rx_tail;
    while (n < cap && tail != rx_head) {
        buf[n++] = rx_ring[tail % LANE_RX_RING];
        tail++;
    }
    rx_tail = tail;
    return n;
}

/* Copy into the transmit ring and return at once: the interrupt feeds the transmitter. Takes what
 * fits under the 12 ms limit; the library offers the rest again at its next turn. */
static size_t lane_write(void *ctx, const uint8_t *buf, size_t len) {
    (void)ctx;
    uint32_t used = tx_head - tx_tail;
    size_t room = used >= lane_tx_limit ? 0 : lane_tx_limit - used;
    size_t n = len < room ? len : room;
    if (n < len) lane_counters[3]++;
    if (n == 0) return 0;
    uint32_t saved = save_and_disable_interrupts();
    uint32_t head = tx_head;
    for (size_t i = 0; i < n; i++) tx_ring[head++ % LANE_TX_RING] = buf[i];
    tx_head = head;
    uart_get_hw(LANE_UART)->imsc |= UART_UARTIMSC_TXIM_BITS; /* the interrupt is asserted while the FIFO is at or below its level */
    restore_interrupts(saved);
    return n;
}

/* ---- klong.h callbacks ------------------------------------------------------------------- */

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
    printf("datagrams: delivered %lu duplicate %lu broken %lu refused %lu resent %lu  ipc in %lu out %lu, inbox overflow %lu  console peer %d, output dropped %lu, with no peer %lu, input dropped %lu, unsolicited %lu\r\n",
           (unsigned long)st.datagrams_delivered, (unsigned long)st.datagrams_duplicate, (unsigned long)st.datagrams_broken, (unsigned long)st.datagrams_refused,
           (unsigned long)st.datagrams_resent, (unsigned long)st.ipc_in, (unsigned long)st.ipc_out, (unsigned long)st.inbox_overflow, (int)st.console_peer, (unsigned long)st.console_dropped, (unsigned long)st.console_no_peer, (unsigned long)st.console_in_dropped, (unsigned long)st.console_unsolicited);
    printf("uart %lu baud: rx ring overruns %lu, fifo overruns %lu, line errors %lu, tx ring full %lu  heap used %lu peak %lu of %lu (cache given back %lu times, %lu bytes)\r\n", (unsigned long)lane_actual_baud,
           (unsigned long)lane_counters[0], (unsigned long)lane_counters[1], (unsigned long)lane_counters[2], (unsigned long)lane_counters[3], (unsigned long)used, (unsigned long)peak, (unsigned long)limit, (unsigned long)kh_drains, (unsigned long)kh_drained_bytes);
    printf("selftest: %s\r\n", st.selftest == 1 ? "pass" : st.selftest == 2 ? "fail" : "running, serving or off");
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
#if KLONG_C_EXAMPLES
static uint64_t c_now_us(void) {
    return time_us_64();
}
#endif
static uint64_t last_drain_ms;
static void k_wait(void *ctx, uint32_t max_ms) {
    (void)ctx;
    /* idle: the cache of freed small blocks goes back to the pool, so that a node at rest holds only what lives (a cached block sits
     * where it was and keeps its neighbours from merging). At most every 20 ms: a call over IPC waits between its messages. */
    if (kh_cached) {
        uint64_t now = time_us_64() / 1000u;
        if (now - last_drain_ms >= 20u) {
            kh_drain();
            last_drain_ms = now;
        }
    }
    /* the lane is read from an interrupt, so a short sleep loses nothing; 200 us keeps the latency of a call low */
    sleep_us(max_ms == 0 ? 0 : 200u);
}
/* Called every turn of the main loop and every 128 calls of an evaluation: the watchdog is fed here, and the canary of the
 * stack looked at. A turn that takes more than the watchdog's 8 s (a primitive over a huge vector) restarts the node. */
static void k_idle(void *ctx) {
    (void)ctx;
    watchdog_update();
    if (!canary_ok()) {
        note_set(NOTE_STACK, 0, 0);
        watchdog_reboot(0, 0, 1);
        for (;;) {
        }
    }
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
    uint32_t a = 0, b = 0;
    memcpy(&a, msg, len < 4 ? len : 4);
    if (len > 4) memcpy(&b, msg + 4, len < 8 ? len - 4 : 4);
    note_set(NOTE_FATAL, a, b);
    if (stdio_usb_connected()) {
        printf("\r\nklong fatal: %.*s\r\n", (int)len, msg);
        stdio_flush();
    }
    watchdog_reboot(0, 0, 100);
    for (;;) tight_loop_contents();
}

#if KLONG_HEAP_DIAG
/* `.heap(mode)` (klong-ffi feature `heapdiag`, given to the runtime with klong_set_diag): the heap as integers. Mode 0 is the summary: the allocator's fields 0 to 20
 * (klong_heap.h) and then the board's, below; modes 1.. are the allocator's map and live blocks (kh_diag). Documented in
 * docs/native/pico-bench.md. */
enum { DIAG_USED = KH_SUMMARY_N, DIAG_PEAK, DIAG_LIMIT, DIAG_RESERVE, DIAG_SDK_BREAK, DIAG_NEWLIB_ARENA, DIAG_NEWLIB_USED, DIAG_NEWLIB_FREE, DIAG_STACK_FREE, DIAG_END, DIAG_STACK_BOTTOM, DIAG_N };
static size_t board_diag(uint32_t mode, int64_t *out, size_t cap) {
    if (mode != 0) return kh_diag(mode, out, cap);
    if (cap < DIAG_N) return 0;
    size_t used, peak, limit;
    kh_diag(0, out, cap);
    klong_heap(&used, &peak, &limit);
    struct mallinfo mi = mallinfo();
    out[DIAG_USED] = (int64_t)used, out[DIAG_PEAK] = (int64_t)peak, out[DIAG_LIMIT] = (int64_t)limit;
    out[DIAG_RESERVE] = KH_RESERVE; /* the runtime keeps this much of the limit back from large requests */
    out[DIAG_SDK_BREAK] = sdk_brk ? sdk_brk - SDK_START : 0;
    out[DIAG_NEWLIB_ARENA] = mi.arena, out[DIAG_NEWLIB_USED] = mi.uordblks, out[DIAG_NEWLIB_FREE] = mi.fordblks;
    out[DIAG_STACK_FREE] = (char *)__builtin_frame_address(0) - STACK_BOTTOM;
    out[DIAG_END] = (int64_t)(uintptr_t)&end, out[DIAG_STACK_BOTTOM] = (int64_t)(uintptr_t)STACK_BOTTOM;
    return DIAG_N;
}
#endif

static void klong_main(void) {
    klong_config cfg;
    klong_lane_config lane;
    memset(&cfg, 0, sizeof cfg);
    cfg.abi_version = KLONG_ABI_VERSION;
    cfg.size = sizeof cfg;
    cfg.alloc = kh_alloc;
    cfg.free = kh_free;
    kh_init(SDK_STOP, STACK_BOTTOM);
    size_t arena = kh_bytes;
    if (arena < KH_TESTED_ARENA) k_fatal(NULL, "heap smaller than the arena tested", 33);
    cfg.grow = kh_grow;
    kh_limits(arena, &cfg.heap_limit, &cfg.heap_reserve);
    cfg.stack_bytes = KLONG_STACK_BUDGET;
    cfg.stack_keep = KLONG_STACK_KEEP;
    cfg.heap_overhead = 0; /* the pool's blocks have no header (klong_heap.h) */
    cfg.heap_round = 8;    /* and are given out in 8s */
    cfg.write = k_write;
    cfg.read = k_read;
    cfg.now_ms = k_now_ms;
    cfg.idle = k_idle;
    cfg.wait = k_wait;
    cfg.reset = k_reset;
    cfg.entropy = k_entropy;
    cfg.fatal = k_fatal;
    cfg.name = "pico-bench";
    cfg.flags = KLONG_FLAG_CRLF; /* files live in RAM (MemFs), in the heap: lost at reset */
    if (klong_init(&cfg) != KLONG_OK) k_fatal(NULL, "klong_init", 10);
#if KLONG_HEAP_DIAG
    klong_set_diag(board_diag);
#endif
#if KLONG_C_EXAMPLES
    if (klong_c_register(c_now_us) != KLONG_OK) k_fatal(NULL, "klong_c_register", 16);
#endif

    memset(&lane, 0, sizeof lane);
    lane.abi_version = KLONG_ABI_VERSION;
    lane.size = sizeof lane;
    lane.lane_read = lane_read;
    lane.lane_write = lane_write;
    lane.node = -1; /* learned from the first frame the router sends to this port */
    lane.seed = (uint32_t)random_bits();
    lane.flags = KLONG_SELFTEST ? KLONG_LANE_SELFTEST : 0;
    lane.baud = LANE_BAUD;
    lane.counters = (const uint32_t *)lane_counters;
    lane.note = boot_note[0] ? boot_note : NULL;
    if (klong_lane_init(&lane) != KLONG_OK) k_fatal(NULL, "klong_lane_init", 15);
    watchdog_enable(8000, true); /* fed in k_idle */
    klong_run();
    for (;;) tight_loop_contents();
}

/* Move the stack pointer to the top of RAM, where the interpreter's stack starts, and call klong_main there: the
 * SDK's own stack is the top 2 KB of the same place, and nothing of it is used after this. */
static void __attribute__((noreturn, noinline)) on_big_stack(void (*fn)(void)) {
    uint32_t top = (uint32_t)&__StackTop & ~7u;
    __asm volatile("msr msp, %0\n\tbx %1\n\t" ::"r"(top), "r"(fn) : "memory");
    for (;;) {
    }
}

int main(void) {
    boot_note_read();
    canary_set();
    /* USB CDC for the console; stdio over uart is off, so uart0 belongs to the lane alone */
    stdio_usb_init();
    lane_init();
    on_big_stack(klong_main);
}
