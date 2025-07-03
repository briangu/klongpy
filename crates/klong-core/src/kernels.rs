use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn simd_add_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let vc = _mm_add_pd(va, vb);
            _mm_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] += b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vc = vaddq_f64(va, vb);
            vst1q_f64(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] += b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

pub fn par_add_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_add_inplace(ac, bc));
}

pub fn simd_sub_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let vc = _mm_sub_pd(va, vb);
            _mm_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] -= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vc = vsubq_f64(va, vb);
            vst1q_f64(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] -= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] -= b[i];
    }
}

pub fn par_sub_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_sub_inplace(ac, bc));
}

pub fn simd_mul_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let vc = _mm_mul_pd(va, vb);
            _mm_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] *= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vc = vmulq_f64(va, vb);
            vst1q_f64(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] *= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

pub fn par_mul_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_mul_inplace(ac, bc));
}

pub fn simd_div_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let vc = _mm_div_pd(va, vb);
            _mm_storeu_pd(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] /= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 2;
        while i < chunks * 2 {
            let va = vld1q_f64(a.as_ptr().add(i));
            let vb = vld1q_f64(b.as_ptr().add(i));
            let vc = vdivq_f64(va, vb);
            vst1q_f64(a.as_mut_ptr().add(i), vc);
            i += 2;
        }
        for j in i..a.len() {
            a[j] /= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] /= b[i];
    }
}

pub fn par_div_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_div_inplace(ac, bc));
}

pub fn simd_add_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] += b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vaddq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] += b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

pub fn par_add_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_add_inplace_f32(ac, bc));
}

pub fn simd_sub_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_sub_ps(va, vb);
            _mm_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] -= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vsubq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] -= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] -= b[i];
    }
}

pub fn par_sub_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_sub_inplace_f32(ac, bc));
}

pub fn simd_mul_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] *= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vmulq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] *= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

pub fn par_mul_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_mul_inplace_f32(ac, bc));
}

pub fn simd_div_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_div_ps(va, vb);
            _mm_storeu_ps(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] /= b[j];
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut i = 0;
        let chunks = a.len() / 4;
        while i < chunks * 4 {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vdivq_f32(va, vb);
            vst1q_f32(a.as_mut_ptr().add(i), vc);
            i += 4;
        }
        for j in i..a.len() {
            a[j] /= b[j];
        }
        return;
    }
    for i in 0..a.len() {
        a[i] /= b[i];
    }
}

pub fn par_div_inplace_f32(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    const CHUNK: usize = 1024;
    a.par_chunks_mut(CHUNK)
        .zip(b.par_chunks(CHUNK))
        .for_each(|(ac, bc)| simd_div_inplace_f32(ac, bc));
}

pub fn map_inplace_f64<F>(a: &mut [f64], mut f: F)
where
    F: FnMut(f64) -> f64,
{
    for v in a.iter_mut() {
        *v = f(*v);
    }
}

pub fn map_inplace_f32<F>(a: &mut [f32], mut f: F)
where
    F: FnMut(f32) -> f32,
{
    for v in a.iter_mut() {
        *v = f(*v);
    }
}
