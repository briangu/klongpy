//! The Objective-C bridge owns Metal objects. A successful operation returns one retained object.
use klong_core::{error::Result, Error};
use std::ffi::{c_char, c_void, CStr};
use std::ptr::{null_mut, NonNull};

extern "C" {
    fn kg_metal_open(
        source: *const u8,
        len: usize,
        ordinal: i64,
        error: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    fn kg_metal_release(object: *mut c_void);
    fn kg_metal_upload(
        context: *mut c_void,
        data: *const c_void,
        bytes: usize,
        error: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    fn kg_metal_download(
        context: *mut c_void,
        buffer: *mut c_void,
        data: *mut c_void,
        bytes: usize,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
    fn kg_metal_dispatch(
        context: *mut c_void,
        kernel: u32,
        a: *mut c_void,
        b: *mut c_void,
        bytes: usize,
        groups: u32,
        width: u32,
        params: *const u32,
        param_bytes: usize,
        error: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
}

pub const BLOCK_WIDTH: u32 = 128;

#[repr(u32)]
pub enum Kernel {
    Elementwise,
    SumBlocks,
    SumColumns,
    GradientTile,
    GradientCollapse,
}

pub struct Object(NonNull<c_void>);

impl Object {
    fn new(create: impl FnOnce(*mut c_char, usize) -> *mut c_void) -> Result<Self> {
        let mut error = [0; 2048];
        let object = create(error.as_mut_ptr(), error.len());
        NonNull::new(object)
            .map(Self)
            .ok_or_else(|| metal_error(&error))
    }

    fn as_ptr(&self) -> *mut c_void {
        self.0.as_ptr()
    }
}

impl Drop for Object {
    fn drop(&mut self) {
        // SAFETY: the bridge returns a retained reference that this wrapper alone releases.
        unsafe { kg_metal_release(self.0.as_ptr()) }
    }
}

fn metal_error(error: &[c_char]) -> Error {
    // SAFETY: the buffer starts zeroed and the bridge writes diagnostics with snprintf.
    let text = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    Error::system(format!("Metal: {text}"))
}

/// # Safety
/// The type must have no padding and accept every bit pattern from Metal.
pub unsafe trait Scalar: Copy {}
// SAFETY: these are primitive lanes without padding or restricted bit patterns.
unsafe impl Scalar for f32 {}
unsafe impl Scalar for i32 {}
unsafe impl Scalar for i64 {}

pub struct Device(Object);

impl Device {
    pub fn open(ordinal: i64) -> Result<Self> {
        let source = include_bytes!("gpu.metal");
        // SAFETY: source and the diagnostic buffer are live throughout this synchronous call.
        Object::new(|error, cap| unsafe {
            kg_metal_open(source.as_ptr(), source.len(), ordinal, error, cap)
        })
        .map(Self)
    }

    pub fn upload<T: Scalar>(&self, data: &[T]) -> Result<Object> {
        let bytes = std::mem::size_of_val(data);
        // SAFETY: Metal copies the borrowed input bytes before returning.
        Object::new(|error, cap| unsafe {
            kg_metal_upload(self.0.as_ptr(), data.as_ptr().cast(), bytes, error, cap)
        })
    }

    pub fn download<T: Scalar>(&self, object: &Object, data: &mut [T]) -> Result<()> {
        let mut error = [0; 2048];
        // SAFETY: the bridge writes exactly the output slice's byte length via memcpy.
        let completed = unsafe {
            kg_metal_download(
                self.0.as_ptr(),
                object.as_ptr(),
                data.as_mut_ptr().cast(),
                std::mem::size_of_val(data),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if completed {
            Ok(())
        } else {
            Err(metal_error(&error))
        }
    }

    pub fn dispatch(
        &self,
        kernel: Kernel,
        a: &Object,
        b: Option<&Object>,
        bytes: usize,
        threads: u32,
        params: &[u32],
    ) -> Result<Object> {
        // SAFETY: the buffers belong to this device and the borrowed parameters are copied by Metal.
        Object::new(|error, cap| unsafe {
            kg_metal_dispatch(
                self.0.as_ptr(),
                kernel as u32,
                a.as_ptr(),
                b.map_or(null_mut(), Object::as_ptr),
                bytes,
                threads.div_ceil(BLOCK_WIDTH),
                BLOCK_WIDTH,
                params.as_ptr(),
                std::mem::size_of_val(params),
                error,
                cap,
            )
        })
    }
}
