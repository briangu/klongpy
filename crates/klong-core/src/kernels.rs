use core::simd::Simd;
use rayon::prelude::*;

pub fn simd_add_inplace(a: &mut [f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f64, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f64, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va + vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f64, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f64, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va - vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f64, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f64, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va * vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f64, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f64, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va / vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f32, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f32, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va + vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f32, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f32, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va - vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f32, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f32, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va * vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
    const LANES: usize = 8;
    let chunks = a.len() / LANES;
    for i in 0..chunks {
        let start = i * LANES;
        let va = Simd::<f32, LANES>::from_slice(&a[start..start + LANES]);
        let vb = Simd::<f32, LANES>::from_slice(&b[start..start + LANES]);
        let vc = va / vb;
        vc.write_to_slice(&mut a[start..start + LANES]);
    }
    for i in chunks * LANES..a.len() {
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
