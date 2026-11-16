# The Klong runtime is a Rust static library built by ../build.sh before this runs:
#   native/target/esp32p4/riscv32imafc-unknown-none-elf/release/libklong_ffi.a
# (rv32imafc, ilp32f: the float ABI of every ESP-IDF build for the ESP32-P4).
get_filename_component(KLONG_NATIVE "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(KLONG_FFI_LIB "${KLONG_NATIVE}/target/esp32p4/riscv32imafc-unknown-none-elf/release/libklong_ffi.a"
    CACHE FILEPATH "libklong_ffi.a built for the ESP32-P4")
if(NOT EXISTS "${KLONG_FFI_LIB}")
    message(FATAL_ERROR "${KLONG_FFI_LIB} is missing: run native/boards/esp32p4/build.sh, which builds it first")
endif()

idf_component_register(SRCS "main.c"
    INCLUDE_DIRS "." "${KLONG_NATIVE}/klong-ffi/include"
    PRIV_REQUIRES esp_driver_uart esp_timer esp_system esp_hw_support heap esp_psram log freertos)

add_prebuilt_library(klong_ffi "${KLONG_FFI_LIB}")
target_link_libraries(${COMPONENT_LIB} PRIVATE klong_ffi)
