//! Device selection for the desktop runtime. CPU is the default.
use klong_core::Interp;
use std::io;

#[cfg(all(target_os = "macos", feature = "metal"))]
#[path = "gpu_runtime.rs"]
mod metal;

pub struct Session {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<metal::Session>,
}

impl Session {
    pub fn open(device: Option<&str>) -> Result<Self, String> {
        let request = device.unwrap_or("cpu");
        let (kind, suffix) = request
            .split_once(':')
            .map_or((request, None), |(kind, index)| (kind, Some(index)));
        match (kind, suffix) {
            ("cpu", None) => Ok(Self {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                metal: None,
            }),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            ("mps", suffix) => {
                let ordinal = match suffix {
                    None => -1,
                    Some(s) if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) => s
                        .parse::<i64>()
                        .map_err(|_| format!("invalid device index in {request}"))?,
                    Some(_) => return Err(format!("invalid device index in {request}")),
                };
                Ok(Self {
                    metal: Some(metal::Session::open(request, ordinal)?),
                })
            }
            #[cfg(all(target_os = "macos", not(feature = "metal")))]
            ("mps", _) => Err("Metal support is disabled in this kg build".to_owned()),
            _ => Err(format!("device {request} is unavailable on this platform")),
        }
    }

    pub fn install(&self, _ip: &mut Interp) {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = &self.metal {
            metal.install(_ip);
        }
    }

    pub fn write_trace(&self) -> io::Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = &self.metal {
            if let Some(path) = std::env::var_os("KLONG_GPU_TRACE") {
                std::fs::write(path, metal.trace())?;
            }
        }
        Ok(())
    }
}
