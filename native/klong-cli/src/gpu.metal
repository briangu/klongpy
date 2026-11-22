#include <metal_stdlib>
using namespace metal;

// The Klong integer lane uses an unsigned representation to get defined wrapping arithmetic.
enum { Float = 0, Int64 = 1, Int32 = 2 };
enum { Add = 0, Subtract = 1, Multiply = 2 };
constant uint block_width = 128;

inline float real_at(device const uchar *data, uint kind, uint i) {
    switch (kind) {
        case Int64: return (float)((device const long *)data)[i];
        case Int32: return (float)((device const int *)data)[i];
        default: return ((device const float *)data)[i];
    }
}

template <typename T>
inline T combine(T a, T b, uint op) {
    switch (op) {
        case Add: return a + b;
        case Subtract: return a - b;
        default: return a * b;
    }
}

kernel void elementwise(device const uchar *a [[buffer(0)]],
                        device const uchar *b [[buffer(1)]],
                        device uchar *out [[buffer(2)]],
                        constant uint *p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    uint x = i / p[1], y = i / p[2];
    uint op = p[3], x_kind = p[4], y_kind = p[5];
    if (x_kind == Int64 && y_kind == Int64) {
        ((device ulong *)out)[i] = combine(((device const ulong *)a)[x], ((device const ulong *)b)[y], op);
    } else if (x_kind == Int32 && y_kind == Int32) {
        ((device uint *)out)[i] = combine(((device const uint *)a)[x], ((device const uint *)b)[y], op);
    } else {
        ((device float *)out)[i] = combine(real_at(a, x_kind, x), real_at(b, y_kind, y), op);
    }
}

template <typename T>
inline void reduce_block(device const uchar *input, device uchar *out,
                         threadgroup T *scratch, uint n, uint lane, uint block) {
    uint i = block * block_width + lane;
    scratch[lane] = i < n ? ((device const T *)input)[i] : T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = block_width / 2; offset > 0; offset /= 2) {
        if (lane < offset) scratch[lane] += scratch[lane + offset];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) ((device T *)out)[block] = scratch[0];
}

kernel void sum_blocks(device const uchar *a [[buffer(0)]],
                       device uchar *out [[buffer(2)]],
                       constant uint *p [[buffer(3)]],
                       uint lane [[thread_index_in_threadgroup]],
                       uint block [[threadgroup_position_in_grid]]) {
    threadgroup ulong scratch[block_width];
    switch (p[1]) {
        case Int64: reduce_block(a, out, scratch, p[0], lane, block); break;
        case Int32: reduce_block(a, out, (threadgroup uint *)scratch, p[0], lane, block); break;
        default: reduce_block(a, out, (threadgroup float *)scratch, p[0], lane, block); break;
    }
}

template <typename T>
inline void reduce_column(device const uchar *input, device uchar *out,
                          uint rows, uint cols, uint col) {
    device const T *a = (device const T *)input;
    T total = a[col];
    for (uint row = 1; row < rows; ++row) total += a[row * cols + col];
    ((device T *)out)[col] = total;
}

kernel void sum_columns(device const uchar *a [[buffer(0)]],
                        device uchar *out [[buffer(2)]],
                        constant uint *p [[buffer(3)]],
                        uint col [[thread_position_in_grid]]) {
    if (col >= p[1]) return;
    switch (p[2]) {
        case Int64: reduce_column<ulong>(a, out, p[0], p[1], col); break;
        case Int32: reduce_column<uint>(a, out, p[0], p[1], col); break;
        default: reduce_column<float>(a, out, p[0], p[1], col); break;
    }
}

kernel void gradient_tile(device const float *a [[buffer(0)]],
                          device float *out [[buffer(2)]],
                          constant uint *p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < p[0]) out[i] = a[i % p[1]];
}

kernel void gradient_collapse(device const float *a [[buffer(0)]],
                              device float *out [[buffer(2)]],
                              constant uint *p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float total = a[i * p[1]];
    for (uint j = 1; j < p[1]; ++j) total += a[i * p[1] + j];
    out[i] = total;
}
