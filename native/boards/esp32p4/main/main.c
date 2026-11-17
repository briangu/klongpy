/*
 * Klong on the ESP32-P4: the serial console of the native runtime on UART0.
 *
 * ESP-IDF supplies boot, the UART driver, the clock and the heap; the runtime is
 * libklong_ffi.a (Rust, native/klong-ffi), reached through klong.h. Everything of Klong
 * runs in one task with a known stack.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_memory_utils.h"
#include "esp_log.h"
#include "esp_random.h"
#include "esp_rom_sys.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "klong.h"

#define CONSOLE_UART UART_NUM_0
#define CONSOLE_BAUD 115200
/* What arrives while the interpreter is busy waits here: 2.8 s of input at full speed. */
#define CONSOLE_RX_BYTES (32 * 1024)

/* The interpreter task: 128 KiB of stack in internal RAM. The runtime is told 120 KiB: the
 * rest is this file's frames above klong_run and room for the UART driver calls below it.
 * (Overridable with -D, as is the heap without PSRAM: a smoke test under QEMU uses a
 * smaller chip.) */
#ifndef KLONG_TASK_STACK
#define KLONG_TASK_STACK (128 * 1024)
#endif
#define KLONG_STACK_BUDGET (KLONG_TASK_STACK - 8 * 1024)
#ifndef KLONG_TASK_CORE
#define KLONG_TASK_CORE 1
#endif
#define KLONG_TASK_PRIORITY 1
#ifndef KLONG_INTERNAL_HEAP
#define KLONG_INTERNAL_HEAP (256 * 1024)
#endif

/* Held back below the heap limit so that running out is a `memory` error at the prompt. */
#define KLONG_HEAP_RESERVE (16 * 1024)

static uint32_t heap_caps; /* where the Klong heap lives: PSRAM, or internal RAM without */

/* Small blocks (what a call of a Klong function allocates and frees: argument frames, boxed values) come from
 * free lists by size class, in slabs of internal RAM. heap_caps_malloc and heap_caps_free take a lock, walk
 * the heap's tables and, in PSRAM, miss the cache on the allocator's own bookkeeping every time: measured on the
 * board, a call of a Klong function took 56 us (about 20,000 cycles at 360 MHz) with three of each per call; a
 * block that goes back to its class costs a few instructions and stays in the fast memory. The runtime says the
 * size to `free`, so no header is needed. Classes of 8 bytes up to 128; a class takes 4 KB slabs from internal RAM
 * (up to SMALL_SLAB_LIMIT of them, then from the heap) and never gives them back. Bigger blocks and bigger
 * alignments go to heap_caps as before; both kinds are freed with heap_caps_free. */
#define SMALL_MAX 128u
#define SMALL_SLAB 4096u
#define SMALL_SLAB_LIMIT (96u * 1024u)
static void *small_free[SMALL_MAX / 8 + 1];
static uint8_t *slab_at, *slab_end;
static size_t slab_total;

static void *small_alloc(size_t size) {
    size_t c = (size + 7u) >> 3;
    void *p = small_free[c];
    if (p) {
        small_free[c] = *(void **)p;
        return p;
    }
    size_t n = c << 3;
    if ((size_t)(slab_end - slab_at) < n) {
        uint8_t *s = slab_total < SMALL_SLAB_LIMIT ? heap_caps_malloc(SMALL_SLAB, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT) : NULL;
        if (!s) return heap_caps_aligned_alloc(8, n, heap_caps); /* internal RAM is spoken for: the heap (8-aligned as a slab block is), freed there below */
        slab_total += SMALL_SLAB;
        slab_at = s;
        slab_end = s + SMALL_SLAB;
    }
    p = slab_at;
    slab_at += n;
    return p;
}

/* 64-bit integers and doubles need 8-byte alignment; heap_caps_malloc gives 4. */
static void *k_alloc(void *ctx, size_t size, size_t align) {
    (void)ctx;
    if (size <= SMALL_MAX && align <= 8) return small_alloc(size ? size : 1);
    if (align <= 4) return heap_caps_malloc(size, heap_caps);
    return heap_caps_aligned_alloc(align, size, heap_caps);
}

static void k_free(void *ctx, void *ptr, size_t size, size_t align) {
    (void)ctx;
    if (!ptr) return;
    if (size <= SMALL_MAX && align <= 8) {
        if (heap_caps & MALLOC_CAP_SPIRAM && esp_ptr_external_ram(ptr)) { /* a small block that came from the heap */
            heap_caps_free(ptr);
            return;
        }
        size_t c = ((size ? size : 1) + 7u) >> 3;
        *(void **)ptr = small_free[c];
        small_free[c] = ptr;
        return;
    }
    heap_caps_free(ptr);
}

/* Raw bytes: the UART driver does no echo, no line buffering and no CR/LF translation
 * (those belong to the VFS console, which this does not use). With no transmit buffer
 * uart_write_bytes returns when everything is in the transmit FIFO. */
static void k_write(void *ctx, const uint8_t *bytes, size_t len) {
    (void)ctx;
    uart_write_bytes(CONSOLE_UART, bytes, len);
}

static size_t k_read(void *ctx, uint8_t *buf, size_t cap) {
    (void)ctx;
    int n = uart_read_bytes(CONSOLE_UART, buf, cap, 0); /* 0 ticks: what is there */
    return n > 0 ? (size_t)n : 0;
}

static uint64_t k_now_ms(void *ctx) {
    (void)ctx;
    return (uint64_t)esp_timer_get_time() / 1000u;
}

static void k_idle(void *ctx) {
    (void)ctx;
    esp_task_wdt_reset();
}

/* Nothing to do: sleep one tick (1 ms) unless input is waiting. */
static void k_wait(void *ctx, uint32_t max_ms) {
    (void)ctx;
    size_t waiting = 0;
    uart_get_buffered_data_len(CONSOLE_UART, &waiting);
    if (waiting == 0 && max_ms > 0) vTaskDelay(1);
}

static void k_reset(void *ctx, int32_t code) {
    (void)ctx, (void)code;
    uart_wait_tx_done(CONSOLE_UART, pdMS_TO_TICKS(200));
    esp_restart();
}

static uint64_t k_entropy(void *ctx) {
    (void)ctx;
    return ((uint64_t)esp_random() << 32) | esp_random();
}

/* A Rust panic: a bug in the runtime. Say so and restart. */
static void k_fatal(void *ctx, const char *msg, size_t len) {
    (void)ctx;
    esp_rom_printf("\r\nklong: fatal: %.*s\r\n", (int)len, msg);
    abort();
}

/* Discard what is in the receive buffer, and wait until nothing more comes. A reset leaves a
 * stray byte on the line (the USB-serial bridge or the boot ROM); read as the first input of
 * the console, it made the first line fail with "input is not valid UTF-8". Flush, wait a
 * short settle time, and repeat while something arrived meanwhile (at most five rounds). */
#define RX_SETTLE_MS 30
static void drain_rx(void) {
    for (int round = 0; round < 5; round++) {
        uart_flush_input(CONSOLE_UART);
        vTaskDelay(pdMS_TO_TICKS(RX_SETTLE_MS));
        size_t waiting = 0;
        uart_get_buffered_data_len(CONSOLE_UART, &waiting);
        if (waiting == 0) return;
    }
    uart_flush_input(CONSOLE_UART);
}

static void klong_task(void *arg) {
    (void)arg;
    esp_task_wdt_add(NULL);

    klong_config cfg = {
        .abi_version = KLONG_ABI_VERSION,
        .size = sizeof cfg,
        .alloc = k_alloc,
        .free = k_free,
        .heap_reserve = KLONG_HEAP_RESERVE,
        .stack_bytes = KLONG_STACK_BUDGET,
        .write = k_write,
        .read = k_read,
        .now_ms = k_now_ms,
        .idle = k_idle,
        .wait = k_wait,
        .reset = k_reset,
        .entropy = k_entropy,
        .fatal = k_fatal,
        .name = "esp32p4",
        .flags = KLONG_FLAG_CRLF,
    };

    /* The heap: three quarters of the free PSRAM (the runtime counts the bytes it asks
     * for; the rest covers the allocator's own overhead). Without PSRAM, 256 KiB of
     * internal RAM: enough for expressions and small programs. */
    size_t psram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    if (psram >= 1024 * 1024) {
        heap_caps = MALLOC_CAP_SPIRAM;
        cfg.heap_limit = psram / 4 * 3;
    } else {
        heap_caps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;
        cfg.heap_limit = KLONG_INTERNAL_HEAP;
    }

    /* (a second time, here: the console starts and prints its banner right after) */
    drain_rx();

    int32_t rc = klong_init(&cfg); /* prints the banner and the first prompt */
    if (rc != KLONG_OK) {
        esp_rom_printf("\r\nklong: klong_init failed (%d)\r\n", (int)rc);
        abort();
    }
    klong_run();
    abort(); /* not reached */
}

void app_main(void) {
    /* From here on nothing but the Klong console may write to UART0: a log line would
     * corrupt the ^A protocol. (The boot log up to this point has already been sent.) */
    esp_log_level_set("*", ESP_LOG_NONE);

    const uart_config_t uart = {
        .baud_rate = CONSOLE_BAUD,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    /* Pins stay as the boot code left them: UART0's defaults, GPIO37 TX and GPIO38 RX. */
    ESP_ERROR_CHECK(uart_driver_install(CONSOLE_UART, CONSOLE_RX_BYTES, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(CONSOLE_UART, &uart));
    uart_wait_tx_done(CONSOLE_UART, pdMS_TO_TICKS(200));
    drain_rx(); /* before the banner is printed and the console starts */

    BaseType_t ok = xTaskCreatePinnedToCore(klong_task, "klong", KLONG_TASK_STACK, NULL, KLONG_TASK_PRIORITY, NULL, KLONG_TASK_CORE);
    if (ok != pdPASS) {
        esp_rom_printf("\r\nklong: cannot create the interpreter task\r\n");
        abort();
    }
}
