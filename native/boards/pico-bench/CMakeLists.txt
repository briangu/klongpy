# Klong on a Raspberry Pi Pico of the 4-Pico bench: the Rust runtime (klong-ffi, feature `lane`)
# as a static library for thumbv6m-none-eabi, and a Pico SDK application around it.
# Build with build.sh (it builds the library first and keeps everything under build/).
cmake_minimum_required(VERSION 3.13)
set(PICO_BOARD pico)
if(NOT DEFINED ENV{PICO_SDK_PATH})
    message(FATAL_ERROR "set PICO_SDK_PATH (build.sh defaults it to ~/scm/pico-sdk)")
endif()
include($ENV{PICO_SDK_PATH}/external/pico_sdk_import.cmake)
project(klong_pico_bench C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

get_filename_component(KLONG_NATIVE "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(KLONG_FFI_LIB "${KLONG_NATIVE}/target/pico-bench/thumbv6m-none-eabi/release/libklong_ffi.a"
    CACHE FILEPATH "libklong_ffi.a built for thumbv6m-none-eabi with --features lane")
if(NOT EXISTS "${KLONG_FFI_LIB}")
    message(FATAL_ERROR "${KLONG_FFI_LIB} is missing: run native/boards/pico-bench/build.sh, which builds it first")
endif()

# -DKLONG_SELFTEST=0 builds the image without the boot self-test; -DLANE_BAUD=115200 for a slower lane
set(KLONG_SELFTEST 1 CACHE STRING "run the boot self-test (1) or not (0)")
set(LANE_BAUD 4500000 CACHE STRING "baud rate of the lane")
set(KLONG_HEAP_DIAG 1 CACHE STRING "the heap diagnostic (.heap) and the allocator table of live blocks: 1 (default) or 0")
# -DKLONG_STACK_KB=96 gives the interpreter a stack of that many KB (default: klong_memory.h, 137), and the heap what is left
set(KLONG_STACK_KB "" CACHE STRING "interpreter stack in KB, empty for the default")
# The flash is read at clk_sys/4 (31 MHz) by the SDK's default second-stage boot loader: a miss of the 16 KB
# instruction cache costs about 60 cycles for 8 bytes. The W25Q16JV on the Pico runs quad reads at 133 MHz: /2 is
# 62.5 MHz. (Set -DPICO_FLASH_CLKDIV=4 to go back.)
set(PICO_FLASH_CLKDIV 2 CACHE STRING "flash clock divider of the boot loader, even")
# The compiled static-profile examples (.c.mavg, .c.pid, .c.fir, .c.check): their C is made by the build from examples/embedded/*.kg, not kept in the
# source tree. The generator uses standard-library Python, `kg compile c`, and standalone `kg` for the expected values.
set(KLONG_C_EXAMPLES 1 CACHE STRING "link the compiled examples (.c.*): 1 (default) or 0")
set(KLONG_PYTHON "python3" CACHE STRING "Python for the example generator (a command with optional arguments); KLONG names the kg binary")
add_executable(klong_pico_bench main.c)
target_compile_definitions(klong_pico_bench PRIVATE KLONG_SELFTEST=${KLONG_SELFTEST} LANE_BAUD=${LANE_BAUD}u KLONG_HEAP_DIAG=${KLONG_HEAP_DIAG} KLONG_C_EXAMPLES=${KLONG_C_EXAMPLES})
if(KLONG_C_EXAMPLES)
    set(KLONG_C_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/klong_c")
    get_filename_component(KLONG_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
    separate_arguments(KLONG_PYTHON_CMD UNIX_COMMAND "${KLONG_PYTHON}")
    file(GLOB KLONG_COMPILER_SOURCES "${KLONG_ROOT}/native/klong-static/src/*.rs" "${KLONG_ROOT}/native/klong-static/klong_rt.h")
    set(KLONG_C_FILES mavg.c pidc.c fir.c klong_c_data.c)
    list(TRANSFORM KLONG_C_FILES PREPEND "${KLONG_C_GEN}/" OUTPUT_VARIABLE KLONG_C_OUTPUTS)
    add_custom_command(
        OUTPUT ${KLONG_C_OUTPUTS} "${KLONG_C_GEN}/klong_c_data.h"
        COMMAND ${KLONG_PYTHON_CMD} "${CMAKE_CURRENT_LIST_DIR}/tools/gen_c_examples.py" "${KLONG_C_GEN}"
        DEPENDS "${CMAKE_CURRENT_LIST_DIR}/tools/gen_c_examples.py" ${KLONG_COMPILER_SOURCES}
                "${KLONG_ROOT}/examples/embedded/mavg.kg" "${KLONG_ROOT}/examples/embedded/mavg.expect"
                "${KLONG_ROOT}/examples/embedded/pid.kg" "${KLONG_ROOT}/examples/embedded/pid.expect"
                "${KLONG_ROOT}/examples/embedded/board.kg"
                "${KLONG_ROOT}/examples/embedded/fir.kg" "${KLONG_ROOT}/examples/embedded/fir.expect"
        COMMENT "generating the C of the compiled examples (kg compile c)"
        VERBATIM)
    # The generated C is held to the runtime header's own standard (C99, pedantic, no warnings); KG_NO_DEFAULT_TRAP: klong_c.c defines kg_trap
    add_library(klong_examples_c OBJECT ${KLONG_C_OUTPUTS})
    set_target_properties(klong_examples_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
    target_include_directories(klong_examples_c PUBLIC "${KLONG_C_GEN}")
    target_compile_definitions(klong_examples_c PRIVATE KG_NO_DEFAULT_TRAP)
    target_compile_options(klong_examples_c PRIVATE -O2 -Wall -Wextra -Werror -pedantic)
    target_sources(klong_pico_bench PRIVATE klong_c.c $<TARGET_OBJECTS:klong_examples_c>)
    target_include_directories(klong_pico_bench PRIVATE "${KLONG_C_GEN}")
endif()
if(NOT "${KLONG_STACK_KB}" STREQUAL "")
    math(EXPR KLONG_STACK_BYTES "${KLONG_STACK_KB} * 1024")
    target_compile_definitions(klong_pico_bench PRIVATE KLONG_STACK_BYTES=${KLONG_STACK_BYTES}u)
endif()
target_include_directories(klong_pico_bench PRIVATE "${KLONG_NATIVE}/klong-ffi/include")
target_compile_options(klong_pico_bench PRIVATE -O2 -Wall -Wextra -Werror)
# the Rust library after the SDK's libraries would let libgcc's helpers win; before them is fine either way
target_link_libraries(klong_pico_bench PRIVATE "${KLONG_FFI_LIB}" pico_stdlib hardware_uart hardware_gpio hardware_irq hardware_watchdog hardware_clocks)
# USB CDC console; the SDK's stdio over a UART stays off so that uart0 (GP16/GP17) is the lane's alone
pico_enable_stdio_usb(klong_pico_bench 1)
pico_enable_stdio_uart(klong_pico_bench 0)
target_compile_definitions(bs2_default PRIVATE PICO_FLASH_SPI_CLKDIV=${PICO_FLASH_CLKDIV})
pico_add_extra_outputs(klong_pico_bench)
