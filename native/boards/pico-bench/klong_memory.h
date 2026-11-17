/*
 * The memory map of the Pico bench image, in one place: main.c uses it and checks it at boot, and the host test of
 * the heap (native/klong-ffi/tests/pico_heap.c) runs the firmware's allocator over an arena of this size.
 *
 * RAM is 264 KB: 256 KB in four banks and 8 KB in two scratch banks above them, all one address range from
 * 0x20000000 to 0x20042000. From the bottom: .data (the vector table, the SDK's RAM functions, the evaluator's
 * hot functions that klong-core's `ram-hot` feature puts there), .bss (the lane's rings and the runtime's static
 * state), the heap, the interpreter's stack, which ends at the top of RAM.
 */
#ifndef KLONG_MEMORY_H
#define KLONG_MEMORY_H

/* The interpreter's stack, from the top of RAM down: 112 KB. A build with another stack (`KLONG_STACK_KB=96 ./build.sh`)
 * has another heap: the bytes between the end of .bss and the bottom of the stack that it gets are printed by build.sh
 * (the whole of RAM is 270,336 bytes less 54,764 of .data and .bss, which is about 215 KB to share). */
#ifndef KLONG_STACK_BYTES
#define KLONG_STACK_BYTES (112u * 1024)
/* The bytes of the runtime's pool that the image built with that stack has at least (the heap of its linker map, 96,720 bytes with the
 * diagnostic on, less KH_SDK_BYTES for newlib's malloc and the alignment; main.c refuses to start with less, and build.sh checks the
 * linker map against it plus KH_SDK_BYTES). The host tests run the allocator over a pool of exactly this size. */
#define KH_TESTED_ARENA 92000u
#elif !defined(KH_TESTED_ARENA)
#define KH_TESTED_ARENA 0u /* another stack: nothing tested at that size */
#endif

#endif
