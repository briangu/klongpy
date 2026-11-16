//! Links and transports (native-runtime.md section 7): how IPC bytes leave the runtime.
//!
//! A [`Transport`] makes links from address strings. A link is a [`StreamLink`] (reliable,
//! ordered bytes: TCP, a pipe) or a [`PacketLink`] (bounded packets to one peer: a UART behind
//! serial framing, a radio, a switched fabric). The runtime puts the packet reliability layer
//! of `klong-ipc` on top of a packet link, after which IPC is the same as over TCP.
//!
//! The runtime has no list of networks: a platform returns its transports from
//! [`Platform::transports`](crate::Platform::transports), and a new network is a `PacketLink`
//! written by whoever owns it. Nothing here blocks.
use alloc::boxed::Box;
use alloc::string::String;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinkError {
    /// The peer closed, or the link was closed here.
    Closed,
    /// Nobody listens at the address.
    Refused,
    /// The address cannot be reached (no route, no such device, a name that does not resolve).
    Unreachable,
    /// The address is not one this transport understands.
    BadAddress,
    /// The address is already in use (listen).
    InUse,
    /// The transport does not do this (a transport that cannot listen).
    Unsupported,
    Io,
}

impl LinkError {
    pub fn text(self) -> &'static str {
        match self {
            LinkError::Closed => "connection closed",
            LinkError::Refused => "connection refused",
            LinkError::Unreachable => "address unreachable",
            LinkError::BadAddress => "bad address",
            LinkError::InUse => "address in use",
            LinkError::Unsupported => "not supported by this transport",
            LinkError::Io => "input/output error",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ListenerId(pub u32);

/// Reliable, ordered bytes.
pub trait StreamLink {
    /// Bytes read; 0 when nothing is waiting. The end of the stream is `Err(Closed)`.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, LinkError>;
    /// Bytes taken; may be fewer than given, 0 when the link is busy.
    fn write(&mut self, buf: &[u8]) -> Result<usize, LinkError>;
    fn close(&mut self);
}

/// Bounded packets to one peer.
pub trait PacketLink {
    /// Bytes per packet, at least 4.
    fn mtu(&self) -> usize;
    /// True when packets are never lost, duplicated or reordered.
    fn reliable(&self) -> bool;
    /// False when the link is busy: offer the same packet again later.
    fn send(&mut self, pkt: &[u8]) -> Result<bool, LinkError>;
    /// One received packet, if any; its length.
    fn recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, LinkError>;
    /// First retransmission timeout, milliseconds.
    fn rto_hint_ms(&self) -> u32;
    fn close(&mut self);
}

pub enum Link {
    Stream(Box<dyn StreamLink>),
    Packet(Box<dyn PacketLink>),
}

pub trait Transport {
    /// The name in an address: `tcp`, `uart`, ...
    fn name(&self) -> &str;
    /// Never blocks for long: a link that is still connecting reads 0 bytes, and a failure
    /// shows on a later read or write.
    fn connect(&mut self, addr: &str) -> Result<Link, LinkError>;
    fn listen(&mut self, addr: &str) -> Result<ListenerId, LinkError>;
    /// A link a peer opened, and a name for the peer.
    fn accept(&mut self, l: ListenerId) -> Option<(Link, String)>;
    fn unlisten(&mut self, l: ListenerId);
}

/// A raw byte pipe with no boundaries and no error detection (a UART) as a packet link:
/// serial framing, COBS and CRC-16 (native-runtime.md section 7.2). A frame that arrives
/// damaged is dropped; the reliability layer above retransmits.
#[cfg(feature = "ipc")]
pub struct SerialLink {
    pipe: Box<dyn StreamLink>,
    deframer: klong_ipc::serial::Deframer,
    mtu: usize,
    rto_ms: u32,
    /// Frame bytes the pipe has not taken yet.
    out: alloc::vec::Vec<u8>,
    sent: usize,
    scratch: alloc::vec::Vec<u8>,
    /// Bytes read from the pipe and not yet deframed.
    inbuf: [u8; 256],
    in_len: usize,
    in_pos: usize,
}

#[cfg(feature = "ipc")]
impl SerialLink {
    /// `mtu` payload bytes per frame (default 240), `rto_ms` the first retransmission timeout
    /// (50 for the reference UART; it should exceed the time two frames take at the baud rate).
    pub fn new(pipe: Box<dyn StreamLink>, mtu: usize, rto_ms: u32) -> SerialLink {
        let mtu = mtu.max(4);
        SerialLink { pipe, deframer: klong_ipc::serial::Deframer::new(mtu), mtu, rto_ms, out: alloc::vec::Vec::new(), sent: 0, scratch: alloc::vec::Vec::new(), inbuf: [0; 256], in_len: 0, in_pos: 0 }
    }

    /// Frames dropped for a bad CRC or bad framing.
    pub fn dropped(&self) -> u32 {
        self.deframer.dropped
    }

    fn flush(&mut self) -> Result<bool, LinkError> {
        while self.sent < self.out.len() {
            let n = self.pipe.write(&self.out[self.sent..])?;
            if n == 0 {
                return Ok(false);
            }
            self.sent += n;
        }
        self.out.clear();
        self.sent = 0;
        Ok(true)
    }
}

#[cfg(feature = "ipc")]
impl PacketLink for SerialLink {
    fn mtu(&self) -> usize {
        self.mtu
    }
    fn reliable(&self) -> bool {
        false
    }
    fn send(&mut self, pkt: &[u8]) -> Result<bool, LinkError> {
        if !self.flush()? {
            return Ok(false);
        }
        if pkt.len() > self.mtu || !klong_ipc::serial::encode_frame(pkt, &mut self.out, &mut self.scratch) {
            self.out.clear();
            return Err(LinkError::Io);
        }
        self.flush()?;
        Ok(true)
    }
    fn recv(&mut self, buf: &mut [u8]) -> Result<Option<usize>, LinkError> {
        loop {
            if self.in_pos == self.in_len {
                self.in_pos = 0;
                self.in_len = self.pipe.read(&mut self.inbuf)?;
                if self.in_len == 0 {
                    return Ok(None);
                }
            }
            let (used, pkt) = self.deframer.push(&self.inbuf[self.in_pos..self.in_len]);
            self.in_pos += used;
            if let Some(p) = pkt {
                if p.len() <= buf.len() {
                    buf[..p.len()].copy_from_slice(p);
                    return Ok(Some(p.len()));
                }
            }
        }
    }
    fn rto_hint_ms(&self) -> u32 {
        self.rto_ms
    }
    fn close(&mut self) {
        self.pipe.close();
    }
}
