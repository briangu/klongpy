/* See klong_c.h. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "klong.h"
#include "klong_c.h"
#include "klong_c_data.h"
#include "fir.h"
#include "mavg.h"
#include "pidc.h"

/* The compiled instances: 32 bytes of RAM in all (sizeof the three structs of the size reports). Everything else is on the stack
 * for the length of a call, or in the runtime's buffer for a result (klong_cfun_register's out_cap): nothing here is sized by the
 * examples' data, so the image's static RAM and heap do not change with them. */
static mavg_t g_mavg;
static pidc_t g_pid;
static fir_t g_fir;
static uint64_t (*g_now)(void);
static int g_trap_code, g_trap_line; /* the first trap of the call that is running */
static kg_float g_duty;              /* the last duty write_pwm was given */

enum { MAVG, PID, FIR, NEX };
#define FIR_IN 256
#define FIR_OUT 241
#define MAX_IN 256
#define ROW 5 /* columns of a row of .c.check() */

/* The tolerance of the compiled results against double-precision ones: the compiled code computes in 32-bit floats (24 bits, a
 * relative step of 6e-8), and a result is a sum of up to 16 products or a short feedback of a few operations, so 1e-5 relative to
 * 1+|value| is about 150 steps. The interpreter and KlongPy both compute in doubles: they must agree to 1e-12. */
#define TOL_FLOAT32 1e-5
#define TOL_DOUBLE 1e-12

/* ---- what the generated code needs of the target ------------------------------------------------------------------------ */

/* kg_trap is built with KG_NO_DEFAULT_TRAP (the default loops for ever): it records the first trap and returns, which the
 * runtime header specifies as memory-safe (the operation gives 0). The wrappers turn a trap into a Klong error. */
void kg_trap(int code, int line) {
    if (g_trap_code == 0) {
        g_trap_code = code;
        g_trap_line = line;
    }
}

/* The externs of pid.kg (read_adc and write_pwm, which board.kg declares as board.adc and board.pwm): constants, as in examples/embedded/pid.expect. */
kg_float read_adc(kg_int channel) {
    (void)channel;
    return (kg_float)0.3;
}
kg_int write_pwm(kg_int channel, kg_float duty) {
    (void)channel;
    g_duty = duty;
    return 0;
}

/* ---- helpers ------------------------------------------------------------------------------------------------------------- */

static int32_t fail(klong_val *out, int32_t code, const char *msg) {
    out->tag = KLONG_T_OTHER;
    out->data = msg;
    return code;
}

static int number(const klong_val *v, double *d) {
    if (v->tag == KLONG_T_INT) *d = (double)v->i;
    else if (v->tag == KLONG_T_FLT) *d = v->f;
    else return 0;
    return 1;
}

/* The example named by a string argument, or -1. */
static int example_of(const klong_val *v) {
    if (v->tag != KLONG_T_STR || v->data == NULL) return -1;
    for (int k = 0; k < NEX; k++)
        if (strlen(klong_c_examples[k].name) == v->len && memcmp(klong_c_examples[k].name, v->data, v->len) == 0) return k;
    return -1;
}

/* The error for an argument that names no example: a type error for a non-string, a range error for an unknown name. */
static int32_t bad_name(const klong_val *v, klong_val *out) {
    if (v->tag != KLONG_T_STR) return fail(out, 1, "the name of an example is a string: \"mavg\", \"pid\" or \"fir\"");
    return fail(out, 2, "the name of an example is \"mavg\", \"pid\" or \"fir\"");
}

static void reset_compiled(int k) {
    switch (k) {
    case MAVG: mavg_init(&g_mavg); break;
    case PID: pidc_init(&g_pid); break;
    default: fir_init(&g_fir); break;
    }
}

/* ---- the functions ------------------------------------------------------------------------------------------------------- */

static int32_t cf_step(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    int pid = (int)(intptr_t)ctx;
    double x;
    (void)n;
    if (!number(&a[0], &x)) return fail(out, 1, pid ? ".c.pid: x must be a number" : ".c.mavg: x must be a number");
    g_trap_code = 0;
    kg_float r = pid ? pidc_step(&g_pid, (kg_float)x) : mavg_step(&g_mavg, (kg_float)x);
    if (g_trap_code) return fail(out, 3, "compiled code trapped (an index out of range or a division by zero)");
    out->tag = KLONG_T_FLT;
    out->f = (double)r;
    return 0;
}

static int32_t cf_fir(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    kg_float in[FIR_IN], res[FIR_OUT];
    double *dst = (double *)out->data; /* the runtime's buffer for FIR_OUT results (out_cap) */
    (void)ctx, (void)n;
    if (a[0].tag != KLONG_T_FLTS && a[0].tag != KLONG_T_INTS) return fail(out, 1, ".c.fir: x must be a list of 256 numbers");
    if (a[0].rank != 1) return fail(out, 1, ".c.fir: x must be a list, not a matrix");
    if (a[0].len != FIR_IN || a[0].data == NULL) return fail(out, 2, ".c.fir: x must have exactly 256 elements");
    if (a[0].tag == KLONG_T_FLTS)
        for (size_t i = 0; i < FIR_IN; i++) in[i] = (kg_float)((const double *)a[0].data)[i];
    else
        for (size_t i = 0; i < FIR_IN; i++) in[i] = (kg_float)((const int64_t *)a[0].data)[i];
    g_trap_code = 0;
    fir_run(&g_fir, in, res);
    if (g_trap_code) return fail(out, 3, "compiled code trapped (an index out of range or a division by zero)");
    for (size_t i = 0; i < FIR_OUT; i++) dst[i] = (double)res[i];
    out->tag = KLONG_T_FLTS;
    out->rank = 1;
    out->len = FIR_OUT;
    return 0;
}

static int32_t cf_reset(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)ctx, (void)n;
    int k = example_of(&a[0]);
    if (k < 0) return bad_name(&a[0], out);
    reset_compiled(k);
    out->tag = KLONG_T_INT;
    out->i = 1;
    return 0;
}

static int32_t cf_src(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)ctx, (void)n;
    int k = example_of(&a[0]);
    if (k < 0) return bad_name(&a[0], out);
    out->tag = KLONG_T_STR;
    out->data = klong_c_examples[k].src;
    out->len = klong_c_examples[k].src_len;
    return 0;
}

/* The interpreter's side of an example: its stand-ins, its source (loaded from flash: functions are kept as the place of their text,
 * nothing is copied to the heap), and its input. Returns what the interpreter returned: 0, or a code and (in *err) its message. */
static int32_t interp_load(const klong_c_example *ex, int with_input, klong_val *err) {
    int32_t c = 0;
    if (ex->prelude[0]) c = klong_eval(ex->prelude, strlen(ex->prelude), err);
    if (c == 0) c = klong_load_static(ex->src, ex->src_len, ex->name, err);
    if (c == 0 && with_input) c = klong_eval(ex->setup, strlen(ex->setup), err);
    return c;
}

static int32_t cf_load(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    klong_val v = {0};
    (void)ctx, (void)n;
    int k = example_of(&a[0]);
    if (k < 0) return bad_name(&a[0], out);
    reset_compiled(k);
    int32_t c = interp_load(&klong_c_examples[k], 0, &v);
    if (c != 0) return fail(out, c, (const char *)v.data);
    out->tag = KLONG_T_INT;
    out->i = 1;
    return 0;
}

static int32_t cf_us(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)ctx, (void)a, (void)n;
    out->tag = KLONG_T_INT;
    out->i = (int64_t)g_now();
    return 0;
}

/* The same stand-ins for the interpreter: the `.extern` lines of board.kg (loaded by pid.kg) resolve to these, registered under their C symbols. */
static int32_t cf_read_adc(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)ctx, (void)a, (void)n;
    out->tag = KLONG_T_FLT;
    out->f = 0.3;
    return 0;
}

static int32_t cf_write_pwm(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    (void)ctx, (void)a, (void)n;
    out->tag = KLONG_T_INT;
    out->i = 0;
    return 0;
}

/* ---- the check ------------------------------------------------------------------------------------------------------------ */

/* got against want within tol of 1+|want|: the relative error in *worst when it is the largest so far; NaN never passes. */
static int close_to(double got, double want, double tol, double *worst) {
    double d = got - want, scale = 1.0 + (want < 0 ? -want : want);
    if (d < 0) d = -d;
    double e = d / scale;
    if (worst) *worst = !(e < 1e9) ? 1e9 : (e > *worst ? e : *worst);
    return e <= tol;
}

static void run_compiled(int k, const kg_float *in, kg_float *res) {
    switch (k) {
    case MAVG:
        mavg_init(&g_mavg);
        for (size_t i = 0; i < klong_c_examples[MAVG].nin; i++) res[i] = mavg_step(&g_mavg, in[i]);
        break;
    case PID:
        pidc_init(&g_pid);
        for (size_t i = 0; i < klong_c_examples[PID].nin; i++) res[i] = pidc_step(&g_pid, in[i]);
        break;
    default:
        fir_init(&g_fir);
        fir_run(&g_fir, in, res);
        break;
    }
}

/* One row: [pass n interp_us compiled_us maxerr_ppb]. interp_us is -1 when the interpreter ran out of memory (the compiled code is
 * then checked against KlongPy's results alone: the table of fir.kg takes more heap than the Pico has) and -2 for any other
 * failure of the interpreter. maxerr_ppb is the largest relative error of the compiled results, in parts per billion of
 * 1+|value|, against the interpreter's results (when it ran), KlongPy's and the values the .expect file pins. Returns 5 when the
 * check was interrupted, else 0. */
static int32_t check_one(int k, int64_t *row) {
    const klong_c_example *ex = &klong_c_examples[k];
    kg_float in[MAX_IN], res[MAX_IN];
    size_t n = ex->nout;
    int ok = 1;
    double worst = 0.0;
    int32_t code = 0;

    for (size_t i = 0; i < ex->nin; i++) in[i] = (kg_float)ex->in[i];
    g_trap_code = 0;
    g_duty = -1;
    uint64_t t0 = g_now();
    run_compiled(k, in, res);
    uint64_t t1 = g_now();
    if (g_trap_code) ok = 0;
    for (size_t i = 0; i < n; i++)
        if (!close_to((double)res[i], ex->ref[i], TOL_FLOAT32, &worst)) ok = 0;
    for (size_t p = 0; p < ex->npin; p++)
        if (!close_to((double)res[ex->pin_idx[p]], ex->pin_val[p], TOL_FLOAT32, &worst)) ok = 0;
    if (k == PID && !(g_duty == res[n - 1])) ok = 0; /* the extern was called with the duty the step returned */

    row[1] = (int64_t)n;
    row[2] = -2;
    klong_val v = {0};
    int32_t c = interp_load(ex, 1, &v);
    if (c == 0) {
        uint64_t t2 = g_now();
        c = klong_eval(ex->expr, strlen(ex->expr), &v);
        uint64_t t3 = g_now();
        if (c == 0) {
            if ((v.tag == KLONG_T_FLTS || v.tag == KLONG_T_INTS) && v.len == n && v.data != NULL) {
                row[2] = (int64_t)(t3 - t2);
                for (size_t i = 0; i < n; i++) {
                    double x = v.tag == KLONG_T_FLTS ? ((const double *)v.data)[i] : (double)((const int64_t *)v.data)[i];
                    if (!close_to(x, ex->ref[i], TOL_DOUBLE, NULL)) ok = 0;
                    if (!close_to((double)res[i], x, TOL_FLOAT32, &worst)) ok = 0;
                }
            } else {
                ok = 0;
            }
        }
    }
    if (c == 4) row[2] = -1; /* out of memory: not a failure of the example */
    else if (c == 5) code = 5;
    else if (c != 0) ok = 0;
    klong_eval(ex->cleanup, strlen(ex->cleanup), NULL);
    klong_eval(".c.in::0", 8, NULL);
    reset_compiled(k);
    row[0] = ok;
    row[3] = (int64_t)(t1 - t0);
    row[4] = (int64_t)(worst * 1e9);
    return code;
}

static int32_t cf_check(void *ctx, const klong_val *a, size_t n, klong_val *out) {
    int64_t *rows = (int64_t *)out->data; /* the runtime's buffer, NEX rows (out_cap) */
    (void)ctx, (void)a, (void)n;
    for (int k = 0; k < NEX; k++) {
        int32_t c = check_one(k, rows + k * ROW);
        if (c != 0) return fail(out, c, "interrupted");
    }
    out->tag = KLONG_T_INTS;
    out->rank = 2;
    out->i = ROW;
    out->len = NEX * ROW;
    return 0;
}

int32_t klong_c_register(uint64_t (*now_us)(void)) {
    static const struct {
        const char *name;
        uint32_t arity;
        klong_cfun fn;
        void *ctx;
        size_t out_cap;
    } fns[] = {
        {".c.mavg", 1, cf_step, (void *)0, 0},
        {".c.pid", 1, cf_step, (void *)1, 0},
        {".c.fir", 1, cf_fir, NULL, FIR_OUT},
        {".c.reset", 1, cf_reset, NULL, 0},
        {".c.src", 1, cf_src, NULL, 0},
        {".c.load", 1, cf_load, NULL, 0},
        {".c.us", 0, cf_us, NULL, 0},
        {".c.check", 0, cf_check, NULL, NEX * ROW},
    };
    /* the externs of board.kg, by C symbol: the programs reach them only through the declarations' signature checks */
    static const struct {
        const char *symbol;
        klong_cfun fn;
        void *ctx;
        size_t out_cap;
    } externs[] = {
        {"read_adc", cf_read_adc, NULL, 0},
        {"write_pwm", cf_write_pwm, NULL, 0},
    };
    g_now = now_us;
    reset_compiled(MAVG);
    reset_compiled(PID);
    reset_compiled(FIR);
    for (size_t i = 0; i < sizeof fns / sizeof fns[0]; i++) {
        int32_t r = klong_cfun_register(fns[i].name, fns[i].arity, fns[i].fn, fns[i].ctx, fns[i].out_cap);
        if (r != KLONG_OK) return r;
    }
    for (size_t i = 0; i < sizeof externs / sizeof externs[0]; i++) {
        int32_t r = klong_extern_register(externs[i].symbol, externs[i].fn, externs[i].ctx, externs[i].out_cap);
        if (r != KLONG_OK) return r;
    }
    return KLONG_OK;
}
