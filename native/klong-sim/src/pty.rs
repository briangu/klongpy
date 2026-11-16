//! A pseudo-terminal pair from the C library, without a crate for it: the board keeps the
//! master end as its "UART", and the slave end is a device path (`/dev/ttys012`,
//! `/dev/pts/3`) that a terminal program or the upload tool opens like a serial port. This
//! module is the one place in the crate that uses `unsafe`.
use std::ffi::CStr;
use std::fs::File;
use std::io;
use std::os::fd::FromRawFd;
use std::os::raw::{c_char, c_int};
use std::os::unix::fs::OpenOptionsExt;

extern "C" {
    fn posix_openpt(flags: c_int) -> c_int;
    fn grantpt(fd: c_int) -> c_int;
    fn unlockpt(fd: c_int) -> c_int;
    fn ptsname(fd: c_int) -> *const c_char;
    fn tcgetattr(fd: c_int, termios: *mut Termios) -> c_int;
    fn tcsetattr(fd: c_int, actions: c_int, termios: *const Termios) -> c_int;
    fn cfmakeraw(termios: *mut Termios);
}

/// Room for the C library's `struct termios` whatever its layout (72 bytes on macOS, 60 on
/// Linux): it is only ever filled and read by the C library.
#[repr(C, align(8))]
struct Termios([u8; 256]);

const O_RDWR: c_int = 2;
#[cfg(target_os = "macos")]
const O_NOCTTY: c_int = 0x20000;
#[cfg(not(target_os = "macos"))]
const O_NOCTTY: c_int = 0x100;

pub struct Pty {
    /// The board's end.
    pub master: File,
    /// The other end, kept open so that the master never reads "hang up" between two users of
    /// the device.
    pub slave: File,
    /// The device to open.
    pub path: String,
}

pub fn open() -> io::Result<Pty> {
    // SAFETY: plain C library calls on a descriptor this function owns; `ptsname` returns a
    // pointer to a static buffer that is copied before anything else is called.
    unsafe {
        let fd = posix_openpt(O_RDWR | O_NOCTTY);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let master = File::from_raw_fd(fd);
        if grantpt(fd) != 0 || unlockpt(fd) != 0 {
            return Err(io::Error::last_os_error());
        }
        let name = ptsname(fd);
        if name.is_null() {
            return Err(io::Error::last_os_error());
        }
        let path = CStr::from_ptr(name).to_string_lossy().into_owned();
        let slave = std::fs::OpenOptions::new().read(true).write(true).custom_flags(O_NOCTTY).open(&path)?;
        // Raw mode: without it the line discipline echoes what the board writes back to the
        // board and waits for line ends.
        let mut t = Termios([0; 256]);
        let sfd = std::os::fd::AsRawFd::as_raw_fd(&slave);
        if tcgetattr(sfd, &mut t) != 0 {
            return Err(io::Error::last_os_error());
        }
        cfmakeraw(&mut t);
        if tcsetattr(sfd, 0, &t) != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(Pty { master, slave, path })
    }
}
