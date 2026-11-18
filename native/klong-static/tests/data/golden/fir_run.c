static void kg_run__1(fir_t *p, const kg_float *a0_x, kg_float *out);

/* run(float[256]) -> float[241] */
static void kg_run__1(fir_t *p, const kg_float *a0_x, kg_float *out) {
    (void)p;
    (void)a0_x;
    /* fir.kg:17: +/taps*x@idx */
    for (kg_int i1 = 0; i1 < 241; i1++) {
        out[i1] = (kg_c_taps[0] * a0_x[kg_c_idx[i1]]);
    }
    for (kg_int i2 = 1; i2 < 16; i2++) {
        for (kg_int i3 = 0; i3 < 241; i3++) {
            out[i3] = (out[i3] + (kg_c_taps[i2] * a0_x[kg_c_idx[((i2 * 241) + i3)]]));
        }
    }
}

void fir_init(fir_t *p) {
    (void)p;
}

size_t fir_size(void) {
    return sizeof(fir_t);
}

void fir_run(fir_t *p, const kg_float *x /* [256] */, kg_float *out /* [241] */) {
    kg_run__1(p, x, out);
}
