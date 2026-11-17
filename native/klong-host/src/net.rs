//! TCP for the host: the `tcp` transport of `.cli` and `.srv`.
//!
//! Sockets are non-blocking once connected. Connecting itself blocks, for at most
//! [`CONNECT_TIMEOUT`]: the standard library has no portable non-blocking connect, and a host
//! program that connects is waiting for the answer anyway.
use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs};
use std::time::Duration;

use klong_rt::link::{Link, LinkError, ListenerId, StreamLink, Transport};

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

fn link_error(e: &std::io::Error) -> LinkError {
    match e.kind() {
        ErrorKind::ConnectionRefused => LinkError::Refused,
        ErrorKind::AddrInUse => LinkError::InUse,
        ErrorKind::ConnectionReset | ErrorKind::ConnectionAborted | ErrorKind::BrokenPipe | ErrorKind::UnexpectedEof | ErrorKind::NotConnected => LinkError::Closed,
        ErrorKind::TimedOut | ErrorKind::AddrNotAvailable => LinkError::Unreachable,
        ErrorKind::InvalidInput => LinkError::BadAddress,
        _ => LinkError::Io,
    }
}

pub struct TcpLink {
    stream: TcpStream,
}

impl TcpLink {
    fn new(stream: TcpStream) -> Result<TcpLink, LinkError> {
        stream.set_nonblocking(true).map_err(|e| link_error(&e))?;
        // a request and its answer are small and wait for each other: no batching
        let _ = stream.set_nodelay(true);
        Ok(TcpLink { stream })
    }
}

impl StreamLink for TcpLink {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, LinkError> {
        loop {
            return match self.stream.read(buf) {
                Ok(0) if !buf.is_empty() => Err(LinkError::Closed),
                Ok(n) => Ok(n),
                Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(0),
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => Err(link_error(&e)),
            };
        }
    }
    fn write(&mut self, buf: &[u8]) -> Result<usize, LinkError> {
        loop {
            return match self.stream.write(buf) {
                Ok(n) => Ok(n),
                Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(0),
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => Err(link_error(&e)),
            };
        }
    }
    fn close(&mut self) {
        let _ = self.stream.shutdown(Shutdown::Both);
    }
}

#[derive(Default)]
pub struct TcpTransport {
    listeners: Vec<(ListenerId, TcpListener)>,
    next: u32,
}

impl TcpTransport {
    pub fn new() -> TcpTransport {
        TcpTransport::default()
    }

    /// The port a listener really has (after listening on port 0).
    pub fn port(&self, l: ListenerId) -> Option<u16> {
        self.listeners.iter().find(|(id, _)| *id == l).and_then(|(_, s)| s.local_addr().ok()).map(|a| a.port())
    }
}

impl Transport for TcpTransport {
    fn name(&self) -> &str {
        "tcp"
    }
    /// `host:port`.
    fn connect(&mut self, addr: &str) -> Result<Link, LinkError> {
        let addrs = addr.to_socket_addrs().map_err(|_| LinkError::Unreachable)?;
        let mut last = LinkError::Unreachable;
        for a in addrs {
            match TcpStream::connect_timeout(&a, CONNECT_TIMEOUT) {
                Ok(s) => {
                    // A connect to a port of the OS's ephemeral range, with nothing listening on that address (the
                    // IPv6 `localhost` before the IPv4 one, say), can be given the same port as its own source
                    // and connect to itself (TCP simultaneous open): the "server" is this socket, which answers
                    // the greeting with the greeting. It is a refused connection.
                    if s.local_addr().ok() == s.peer_addr().ok() {
                        last = LinkError::Refused;
                        continue;
                    }
                    return Ok(Link::Stream(Box::new(TcpLink::new(s)?)));
                }
                Err(e) => last = link_error(&e),
            }
        }
        Err(last)
    }
    /// `bind:port`; an empty bind address is every interface, as KlongPy's `.srv(port)`.
    fn listen(&mut self, addr: &str) -> Result<ListenerId, LinkError> {
        let full = if addr.starts_with(':') { format!("0.0.0.0{}", addr) } else { String::from(addr) };
        let l = TcpListener::bind(full.as_str()).map_err(|e| link_error(&e))?;
        l.set_nonblocking(true).map_err(|e| link_error(&e))?;
        self.next += 1;
        let id = ListenerId(self.next);
        self.listeners.push((id, l));
        Ok(id)
    }
    fn accept(&mut self, l: ListenerId) -> Option<(Link, String)> {
        let (_, s) = self.listeners.iter().find(|(id, _)| *id == l)?;
        let (stream, peer) = s.accept().ok()?;
        let link = TcpLink::new(stream).ok()?;
        Some((Link::Stream(Box::new(link)), peer.to_string()))
    }
    fn unlisten(&mut self, l: ListenerId) {
        self.listeners.retain(|(id, _)| *id != l);
    }
}
