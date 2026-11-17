/*
 * klong_rt.h - the run-time support of C emitted by KlongPy (kgpy --emit-c).
 *
 * Header only, C99, no board, chip or vendor. It holds:
 *
 *   kg_int, kg_float       the two element types (KG_INT64 and KG_FLOAT64 widen them;
 *                          the emitted <prog>.h defines those the program was compiled for)
 *   kg_trap(code, line)    called on a run-time error; the default (weak) never returns
 *   KG_BOUNDS              1 (default): an index out of range calls kg_trap; 0: unchecked
 *   KG_DIV0                ieee (default): x%0 is inf or nan; trap: it calls kg_trap
 *   kg_iadd ...            integer arithmetic that wraps at the width of kg_int
 *   kg_idiv, kg_imod       integer divide and remainder (truncating; a zero divisor traps)
 *   kg_fmod, kg_floor, kg_pow, kg_fmin, kg_fmax, kg_match
 *
 * With KG_FREESTANDING defined nothing from libc is used: kg_pow and kg_fmod
 * are computed here, and memcpy and memset are defined (weak), because a
 * compiler may emit calls to them for plain loops. KG_NO_MEM_BUILTINS leaves
 * those two out when the target already provides them.
 *
 * Without KG_FREESTANDING the only library function used is pow (powf for a
 * 32-bit kg_float), and only by programs that use ^ with a float result (the
 * emitted header then defines KG_NEED_MATH and <math.h> is included).
 */
#ifndef KLONG_RT_H
#define KLONG_RT_H

#include <stddef.h>
#include <stdint.h>

#ifdef KG_INT64
typedef int64_t kg_int;
typedef uint64_t kg_uint;
#define KG_INT_MAX INT64_MAX
#define KG_INT_MIN INT64_MIN
#define KG_INT_LIMIT 9223372036854775808.0 /* 2^63 */
#else
typedef int32_t kg_int;
typedef uint32_t kg_uint;
#define KG_INT_MAX INT32_MAX
#define KG_INT_MIN INT32_MIN
#define KG_INT_LIMIT 2147483648.0 /* 2^31 */
#endif

#ifdef KG_FLOAT64
typedef double kg_float;
#else
typedef float kg_float;
#endif

#if defined(__GNUC__) || defined(__clang__)
#define KG_WEAK __attribute__((weak))
#if defined(KG_TRAP_FLAG)
/* hardware (kgpy --emit-hls): the helpers are always inlined, so that the tool sees constant widths and shifts and
 * does not build one general module (two barrel shifters) per call */
#define KG_UNUSED __attribute__((unused, always_inline))
#else
#define KG_UNUSED __attribute__((unused))
#endif
#define KG_INF ((kg_float)__builtin_inf())
#define KG_NAN ((kg_float)__builtin_nan(""))
#else
#define KG_WEAK
#define KG_UNUSED
#define KG_INF ((kg_float)(1.0e300 * 1.0e300))
#define KG_NAN ((kg_float)(KG_INF - KG_INF))
#endif

/* ---- policies ----------------------------------------------------------- */

#ifndef KG_BOUNDS
#define KG_BOUNDS 1
#endif

#define KG_DIV0_ieee 0
#define KG_DIV0_trap 1
#ifndef KG_DIV0
#define KG_DIV0 ieee
#endif
#define KG_CAT_(a, b) a##b
#define KG_CAT(a, b) KG_CAT_(a, b)
#define KG_DIV0_TRAPS (KG_CAT(KG_DIV0_, KG_DIV0) == 1)

/* ---- traps -------------------------------------------------------------- */

#define KG_TRAP_BOUNDS 1 /* an index out of range (KG_BOUNDS) */
#define KG_TRAP_DIV0 2   /* an integer division by zero, or x%0 with KG_DIV0=trap */

/*
 * Called with one of the codes above and the line of the Klong source. The
 * default (a weak definition in each generated <prog>.c, which defines
 * KG_RT_IMPLEMENTATION before it includes this header) loops forever; a
 * target defines its own in any other file to reset, log or longjmp.
 * KG_NO_DEFAULT_TRAP leaves the default out. If it returns, the operation that trapped gives 0 (an index
 * out of range reads or writes item 0), so memory stays safe; the values
 * computed after a trap are unspecified.
 */
void kg_trap(int code, int line);

/*
 * KG_TRAP_FLAG (hardware, kgpy --emit-hls): define it to the name of a counter. The generated
 * <prog>.c then defines that counter and a kg_trap that sets bit <code> of it and returns, in
 * place of the default below (KG_TRAP_BOUNDS is bit 1, KG_TRAP_DIV0 bit 2); the caller reads and clears it.
 */
#if defined(KG_RT_IMPLEMENTATION) && defined(KG_TRAP_FLAG)
unsigned KG_TRAP_FLAG;
void kg_trap(int code, int line) {
    (void)line;
    KG_TRAP_FLAG |= 1u << (unsigned)code;
}
#elif defined(KG_RT_IMPLEMENTATION) && !defined(KG_NO_DEFAULT_TRAP) && (defined(__GNUC__) || defined(__clang__))
KG_WEAK void kg_trap(int code, int line) {
    (void)code;
    (void)line;
    for (;;) {
    }
}
#endif

/* ---- freestanding memcpy and memset -------------------------------------- */

#if defined(KG_RT_IMPLEMENTATION) && defined(KG_FREESTANDING) && !defined(KG_NO_MEM_BUILTINS)
void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);
/* The empty asm statement keeps the compiler from turning the loop back into a call of the function itself. */
#if defined(__GNUC__) || defined(__clang__)
#define KG_NO_IDIOM() __asm__("")
#else
#define KG_NO_IDIOM() ((void)0)
#endif
KG_WEAK void *memcpy(void *dst, const void *src, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) {
        *d++ = *s++;
        KG_NO_IDIOM();
    }
    return dst;
}
KG_WEAK void *memset(void *dst, int c, size_t n) {
    unsigned char *d = (unsigned char *)dst;
    while (n--) {
        *d++ = (unsigned char)c;
        KG_NO_IDIOM();
    }
    return dst;
}
#endif

/* ---- indices -------------------------------------------------------------- */

/* An index into n items: a negative index counts from the end; out of range traps (KG_BOUNDS). */
static inline KG_UNUSED kg_int kg_ix(kg_int i, kg_int n, int line) {
    if (i < 0) {
        i = (kg_int)((kg_uint)i + (kg_uint)n);
    }
#if KG_BOUNDS
    if (i < 0 || i >= n) {
        kg_trap(KG_TRAP_BOUNDS, line);
        return 0;
    }
#else
    (void)line;
#endif
    return i;
}

/* The rotation count k of n:+a reduced to 0 .. n-1. */
static inline KG_UNUSED kg_int kg_rot(kg_int k, kg_int n) {
    kg_int r = k % n;
    return r < 0 ? r + n : r;
}

/* ---- integers: arithmetic wraps at the width of kg_int -------------------- */

static inline KG_UNUSED kg_int kg_iadd(kg_int a, kg_int b) { return (kg_int)((kg_uint)a + (kg_uint)b); }
static inline KG_UNUSED kg_int kg_isub(kg_int a, kg_int b) { return (kg_int)((kg_uint)a - (kg_uint)b); }
static inline KG_UNUSED kg_int kg_imul(kg_int a, kg_int b) { return (kg_int)((kg_uint)a * (kg_uint)b); }
static inline KG_UNUSED kg_int kg_ineg(kg_int a) { return (kg_int)((kg_uint)0 - (kg_uint)a); }
static inline KG_UNUSED kg_int kg_imax(kg_int a, kg_int b) { return a > b ? a : b; }
static inline KG_UNUSED kg_int kg_imin(kg_int a, kg_int b) { return a < b ? a : b; }

static inline KG_UNUSED kg_int kg_ilt(kg_int a, kg_int b) { return a < b; }
static inline KG_UNUSED kg_int kg_igt(kg_int a, kg_int b) { return a > b; }
static inline KG_UNUSED kg_int kg_ieq(kg_int a, kg_int b) { return a == b; }

/* a^n for a constant n >= 0, wrapping. */
static inline KG_UNUSED kg_int kg_ipow(kg_int a, kg_int n) {
    kg_uint r = 1, b = (kg_uint)a;
    while (n > 0) {
        if (n & 1) {
            r *= b;
        }
        b *= b;
        n >>= 1;
    }
    return (kg_int)r;
}

/* a:%b truncates toward zero; a zero divisor always traps. */
static inline KG_UNUSED kg_int kg_idiv(kg_int a, kg_int b, int line) {
    if (b == 0) {
        kg_trap(KG_TRAP_DIV0, line);
        return 0;
    }
    if (b == -1) {
        return kg_ineg(a); /* KG_INT_MIN / -1 overflows in C */
    }
    return a / b;
}

/* a!b has the sign of a; a zero divisor always traps. */
static inline KG_UNUSED kg_int kg_imod(kg_int a, kg_int b, int line) {
    if (b == 0) {
        kg_trap(KG_TRAP_DIV0, line);
        return 0;
    }
    if (b == -1) {
        return 0;
    }
    return a % b;
}

/* ---- floats ------------------------------------------------------------------ */

static inline KG_UNUSED kg_float kg_fabs(kg_float x) { return x < 0 ? -x : x; }
static inline KG_UNUSED kg_int kg_flt(kg_float a, kg_float b) { return a < b; }
static inline KG_UNUSED kg_int kg_fgt(kg_float a, kg_float b) { return a > b; }
static inline KG_UNUSED kg_int kg_feq(kg_float a, kg_float b) { return a == b; }

/* a%b */
static inline KG_UNUSED kg_float kg_fdiv(kg_float a, kg_float b, int line) {
#if KG_DIV0_TRAPS
    if (b == 0) {
        kg_trap(KG_TRAP_DIV0, line);
        return 0;
    }
#else
    (void)line;
#endif
    return a / b;
}

/* _x: toward minus infinity, saturating at the range of kg_int; nan gives 0. */
static inline KG_UNUSED kg_int kg_floor(kg_float x) {
    kg_int i;
    if (x != x) {
        return 0;
    }
    if (x >= (kg_float)KG_INT_LIMIT) {
        return KG_INT_MAX;
    }
    if (x < -(kg_float)KG_INT_LIMIT) {
        return KG_INT_MIN;
    }
    i = (kg_int)x;
    if ((kg_float)i > x) {
        i--;
    }
    return i;
}

/* a|b and a&b: nan if either is nan; -0.0 is below 0.0. */
static inline KG_UNUSED kg_float kg_fmax(kg_float a, kg_float b) {
    if (a != a) {
        return a;
    }
    if (b != b) {
        return b;
    }
    if (a > b) {
        return a;
    }
    if (b > a) {
        return b;
    }
    return a == 0 ? a + b : a;
}

static inline KG_UNUSED kg_float kg_fmin(kg_float a, kg_float b) {
    if (a != a) {
        return a;
    }
    if (b != b) {
        return b;
    }
    if (a < b) {
        return a;
    }
    if (b < a) {
        return b;
    }
    return a == 0 ? -((-a) + (-b)) : a;
}

/* a~b for two numbers: |a-b| <= 1e-8 + 1e-5*|b| (KlongPy uses numpy.isclose); nan matches nothing. */
static inline KG_UNUSED kg_int kg_match(kg_float a, kg_float b) {
    if (a == b) {
        return 1;
    }
    if (a != a || b != b || a - a != 0 || b - b != 0) {
        return 0; /* a nan, or an infinity that is not equal */
    }
    return kg_fabs(a - b) <= (kg_float)1e-8 + (kg_float)1e-5 * kg_fabs(b);
}

#if defined(KG_NEED_MATH) && !defined(KG_FREESTANDING)
#include <math.h>
#ifdef KG_FLOAT64
#define KG_LIBM_POW pow
#else
#define KG_LIBM_POW powf
#endif
#endif

/*
 * a!b for floats: the remainder of truncating division, with the sign of a
 * (C's fmod). Computed by exact subtraction, so no library is needed.
 */
static inline KG_UNUSED kg_float kg_fmod(kg_float a, kg_float b, int line) {
    kg_float x = kg_fabs(a), y = kg_fabs(b), t;
#if KG_DIV0_TRAPS
    if (b == 0) {
        kg_trap(KG_TRAP_DIV0, line);
        return 0;
    }
#else
    (void)line;
#endif
    if (a != a || b != b || y == 0 || x - x != 0) {
        return KG_NAN;
    }
    if (y - y != 0 || x < y) {
        return a;
    }
    while (x >= y) {
        t = y;
        while (t + t <= x) {
            t = t + t;
        }
        x -= t;
    }
    return a < 0 ? -x : x;
}

#if !defined(KG_LIBM_POW)
/* Freestanding exp and log for kg_pow: range reduction and a short series, good to a few ulp of a float. */
static inline KG_UNUSED kg_float kg_log_(kg_float x) {
    /* x > 0: x = m * 2^e with m in [sqrt(1/2), sqrt(2)); log m by the atanh series */
    kg_float e = 0, s, s2, term, sum;
    int k;
    while (x >= (kg_float)1.4142135623730951) {
        x *= (kg_float)0.5;
        e += 1;
    }
    while (x < (kg_float)0.7071067811865476) {
        x *= 2;
        e -= 1;
    }
    s = (x - 1) / (x + 1);
    s2 = s * s;
    term = s;
    sum = 0;
    for (k = 1; k < 40; k += 2) {
        sum += term / (kg_float)k;
        term *= s2;
    }
    return 2 * sum + e * (kg_float)0.6931471805599453;
}

static inline KG_UNUSED kg_float kg_exp_(kg_float x) {
    /* e^x = 2^n * e^r, r in [-ln2/2, ln2/2] */
    kg_float n, r, term, sum, p;
    int k, i;
    if (x > 710) {
        return KG_INF;
    }
    if (x < -750) {
        return 0;
    }
    n = (kg_float)kg_floor(x / (kg_float)0.6931471805599453 + (kg_float)0.5);
    r = x - n * (kg_float)0.6931471805599453;
    term = 1;
    sum = 1;
    for (k = 1; k < 24; k++) {
        term *= r / (kg_float)k;
        sum += term;
    }
    p = n < 0 ? (kg_float)0.5 : 2;
    i = (int)(n < 0 ? -n : n);
    while (i-- > 0) {
        sum *= p;
    }
    return sum;
}
#endif

/* a^b with a float result. */
static inline KG_UNUSED kg_float kg_pow(kg_float a, kg_float b) {
#if defined(KG_LIBM_POW)
    return KG_LIBM_POW(a, b);
#else
    kg_float r = 1, base;
    kg_int n;
    if (b == 0 || a == 1) {
        return 1;
    }
    if (a != a || b != b) {
        return KG_NAN;
    }
    if (b == (kg_float)(kg_int)kg_floor(b) && kg_fabs(b) < 65536) {
        /* a whole exponent: repeated squaring, exact for small results */
        n = kg_floor(kg_fabs(b));
        base = a;
        while (n > 0) {
            if (n & 1) {
                r *= base;
            }
            base *= base;
            n >>= 1;
        }
        return b < 0 ? 1 / r : r;
    }
    if (a < 0) {
        return KG_NAN;
    }
    if (a == 0) {
        return b > 0 ? 0 : KG_INF;
    }
    if (a - a != 0) {
        return b > 0 ? a : 0;
    }
    return kg_exp_(b * kg_log_(a));
#endif
}


/* ---- fixed-point: fixed<W,I> is a W-bit two's complement integer (stored in an int32_t or int64_t) ----
 * whose value is raw * 2^-F, F = W - I. Results wrap to W bits; multiply and divide round toward minus
 * infinity. Everything is integer arithmetic (no __int128, no library): the same code is the reference of the
 * hardware flow, and the tests hold it equal to klongpy/static/fixed.py bit for bit. */

/* an arithmetic shift right that is defined for negative x */
static inline KG_UNUSED int64_t kg_sar64(int64_t x, int n) {
    if (n == 0) {
        return x;
    }
    return x >= 0 ? (int64_t)((uint64_t)x >> n) : (int64_t)(~((~(uint64_t)x) >> n));
}

/* v wrapped to w bits (2 <= w <= 64) */
static inline KG_UNUSED int64_t kg_fxwrap(int64_t v, int w) {
    if (w >= 64) {
        return v;
    }
    return kg_sar64((int64_t)((uint64_t)v << (64 - w)), 64 - w);
}

static inline KG_UNUSED int64_t kg_fxadd(int64_t a, int64_t b, int w) { return kg_fxwrap((int64_t)((uint64_t)a + (uint64_t)b), w); }
static inline KG_UNUSED int64_t kg_fxsub(int64_t a, int64_t b, int w) { return kg_fxwrap((int64_t)((uint64_t)a - (uint64_t)b), w); }
static inline KG_UNUSED int64_t kg_fxneg(int64_t a, int w) { return kg_fxwrap((int64_t)((uint64_t)0 - (uint64_t)a), w); }

/* floor(a * b / 2^f) wrapped; the product of two operands of at most 32 bits fits 64 bits */
static inline KG_UNUSED int64_t kg_fxmul(int64_t a, int64_t b, int f, int w) { return kg_fxwrap(kg_sar64(a * b, f), w); }

/* the same for up to 64 bits: the 128-bit product from 32-bit limbs, then the low 64 bits of its shift */
static inline KG_UNUSED int64_t kg_fxmul128(int64_t a, int64_t b, int f, int w) {
    uint64_t ua = (uint64_t)a, ub = (uint64_t)b;
    uint64_t a0 = ua & 0xffffffffu, a1 = ua >> 32, b0 = ub & 0xffffffffu, b1 = ub >> 32;
    uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
    uint64_t lo = (p00 & 0xffffffffu) | (mid << 32);
    uint64_t hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    uint64_t r;
    if (a < 0) {
        hi -= ub;
    }
    if (b < 0) {
        hi -= ua;
    }
    r = f == 0 ? lo : ((lo >> f) | (hi << (64 - f)));
    return kg_fxwrap((int64_t)r, w);
}

/* floor(a * 2^f / b) wrapped, for w <= 32; a division by zero traps */
static inline KG_UNUSED int64_t kg_fxdiv(int64_t a, int64_t b, int f, int w, int line) {
    int64_t n, q;
    if (b == 0) {
        kg_trap(KG_TRAP_DIV0, line);
        return 0;
    }
    n = (int64_t)((uint64_t)a << f);
    q = n / b;
    if (n % b != 0 && ((n < 0) != (b < 0))) {
        q -= 1;
    }
    return kg_fxwrap(q, w);
}

/* int to fixed: a * 2^f wrapped */
static inline KG_UNUSED int64_t kg_fxfromi(int64_t a, int f, int w) { return kg_fxwrap((int64_t)((uint64_t)a << f), w); }

/* fixed to another fixed type: the change of fraction bits d = F(dst) - F(src) */
static inline KG_UNUSED int64_t kg_fxshift(int64_t a, int d, int w) {
    return kg_fxwrap(d >= 0 ? (int64_t)((uint64_t)a << d) : kg_sar64(a, -d), w);
}

/* float to fixed: floor(x * 2^f) wrapped; nan and anything beyond a 64-bit integer give 0 */
static inline KG_UNUSED int64_t kg_fxfromf(double x, double scale, int w) {
    double s = x * scale, t;
    if (!(s >= -9223372036854775808.0 && s < 9223372036854775808.0)) {
        return 0;
    }
    t = (double)(int64_t)s;
    if (t > s) {
        t -= 1.0;
    }
    return kg_fxwrap((int64_t)t, w);
}

/* _x: floor to an int (wrapped to the width of kg_int) */
static inline KG_UNUSED kg_int kg_fxfloor(int64_t a, int f) { return (kg_int)(kg_uint)(uint64_t)kg_sar64(a, f); }

#endif /* KLONG_RT_H */
