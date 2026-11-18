/*
 * The compiled examples of the static profile (examples/embedded/{mavg,pid,fir}.kg) as system functions of the board, and the check
 * that runs each of them against the interpreter. The C of the examples is generated at build time (tools/gen_c_examples.py), so the
 * firmware has no copy of it in the source tree; this file and klong_c.c are all that is written by hand, and nothing in them knows
 * about the Pico: the board gives a microsecond clock, and the host test (tests/static/test_pico_check.py) links the same two files
 * into a program on the desktop.
 *
 * System functions (the `.c.` prefix is the board's; `.entry` and `.extern` are KlongPy's declarations of a static-profile file,
 * accepted and ignored so that the interpreter can load the same text):
 *
 *   .c.mavg(x)    one step of mavg.kg (float -> float), the compiled instance keeps the window
 *   .c.pid(x)     one step of pid.kg (float -> float); the externs are stand-ins: read.adc gives 0.3, write.pwm remembers the duty
 *   .c.fir(x)     fir.kg's run: a list of 256 numbers to a list of 241 floats
 *   .c.reset(n)   n is "mavg", "pid" or "fir": the compiled instance starts again
 *   .c.src(n)     the Klong source of that example, as a string
 *   .c.load(n)    reset the compiled instance and load the Klong source into the interpreter (it defines that program's
 *                 globals: buf pos step | kp ki kd dt lo hi integ prev clamp step read.adc write.pwm | raw taps idx run)
 *   .c.us()       the board's clock in microseconds
 *   .c.check()    run every example both ways on fixed inputs; one row per example: [pass n interp_us compiled_us maxerr_ppb]
 *
 * An argument of the wrong type or length is a Klong error (type or range), never a crash; a trap of the compiled code (an index
 * out of range, which the examples cannot reach) is a system error.
 */
#ifndef KLONG_C_H
#define KLONG_C_H

#include <stdint.h>

/* Define the functions above. Call once, after klong_init. `now_us` is a monotonic clock in microseconds. Returns KLONG_OK or the
 * error of klong_cfun_register. */
int32_t klong_c_register(uint64_t (*now_us)(void));

#endif
