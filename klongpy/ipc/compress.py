"""kdb+ IPC message compression (reference section 5).

Both directions are ports of the published Java (``javakdb`` ``c.java``) and
JavaScript (``c.js``) client code, as transcribed in the reference
(``docs/ipc_capabilities.md`` 5.2 to 5.4).  Neither has been checked against
a message produced by a real kdb+ process (VERIFY item 3 of the design).

Both functions work on whole messages including the 8-byte header:

``decompress(message) -> message``
    takes a message whose header byte 2 is 1 and returns the equivalent
    uncompressed message (header byte 2 cleared, length field rewritten).
``compress(message) -> message``
    returns the compressed form when kdb+'s size rules say so (more than 2000
    bytes and the result is under half), otherwise the input unchanged.

Safety: the declared uncompressed size is checked against ``max_size`` before
any allocation, against an amplification bound derived from the stream
format, and every back-reference is validated (``r >= 8``, ``r < s-1``,
``n+2 <= usize-s``), as the Java client does.  Malformed input raises
:class:`ProtocolError` and nothing else.
"""
import struct

from .errors import ProtocolError

#: default upper bound for a (decompressed) message, bytes
MAX_MESSAGE = 256 * 1024 * 1024

#: kdb+ compresses only messages larger than this (reference 5.1)
COMPRESS_THRESHOLD = 2000

# Largest expansion of the stream format: a full group (1 flag byte, 8
# back-references of 2 bytes) of 17 input bytes produces 8 * 257 = 2056 output
# bytes, a ratio of about 121.  Anything claiming more is corrupt or hostile,
# and rejecting it before allocating bounds memory by the input size, not by
# the attacker's declared size.  (Not in the reference; own addition.)
_MAX_EXPANSION = 125


def decompress(data, max_size=MAX_MESSAGE):
    """Return the uncompressed message for the compressed message ``data``."""
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    n = len(data)
    if n < 12:
        raise ProtocolError("compressed message shorter than its 12-byte header")
    if data[0] > 1:
        raise ProtocolError("bad byte-order flag in message header")
    order = "<" if data[0] == 1 else ">"
    total, usize = struct.unpack_from(order + "II", data, 4)
    if total != n:
        raise ProtocolError("compressed length field does not match the message")
    if usize < 8 or usize > max_size or usize > 0x7FFFFFFF:
        raise ProtocolError("declared uncompressed size %d out of range" % usize)
    if usize - 8 > _MAX_EXPANSION * (n - 12):
        raise ProtocolError("declared uncompressed size impossible for the compressed length")

    dst = bytearray(usize)
    aa = [0] * 256
    s = 8          # write position in dst
    p = 8          # next position whose pair is not hashed yet
    d = 12         # read position in data
    i = 0          # current flag bit, 0 = need a new flag byte
    f = 0
    while s < usize:
        if i == 0:
            if d >= n:
                raise ProtocolError("compressed stream truncated")
            f = data[d]
            d += 1
            i = 1
        backref = (f & i) != 0
        if backref:
            if d + 2 > n:
                raise ProtocolError("compressed stream truncated")
            r = aa[data[d]]
            ext = data[d + 1]
            d += 2
            if r < 8 or r >= s - 1 or ext + 2 > usize - s:
                raise ProtocolError("invalid back-reference in compressed stream")
            dst[s] = dst[r]
            dst[s + 1] = dst[r + 1]
            s += 2
            r += 2
            if ext:
                # forward byte-by-byte copy whose source may overlap the
                # destination: repeat the gap pattern.
                gap = s - r
                if gap >= ext:
                    dst[s:s + ext] = dst[r:r + ext]
                else:
                    pat = bytes(dst[r:s])
                    dst[s:s + ext] = (pat * (ext // gap + 1))[:ext]
        else:
            if d >= n:
                raise ProtocolError("compressed stream truncated")
            dst[s] = data[d]
            s += 1
            d += 1
        # index every adjacent pair written so far (before skipping the run)
        while p < s - 1:
            aa[dst[p] ^ dst[p + 1]] = p
            p += 1
        if backref:
            s += ext
            p = s
        i <<= 1
        if i == 256:
            i = 0

    dst[0] = data[0]
    dst[1] = data[1]
    dst[2] = 0
    dst[3] = data[3]
    struct.pack_into(order + "I", dst, 4, usize)
    return bytes(dst)


def compress(message, threshold=COMPRESS_THRESHOLD):
    """Return ``message`` compressed, or unchanged if it is not worth it.

    ``message`` is a whole uncompressed message with its 8-byte header.  The
    result is compressed only if the message is longer than 2000 bytes and the
    compressed form is shorter than half of it (kdb+'s rule).  ``threshold`` is
    exposed for tests only.
    """
    y = message if isinstance(message, (bytes, bytearray)) else bytes(message)
    t = len(y)
    if t <= threshold:
        return bytes(y)
    if t < 8 or y[0] > 1 or y[2] != 0:
        raise ProtocolError("not an uncompressed message")
    order = "<" if y[0] == 1 else ">"
    e = t // 2
    out = bytearray(e)
    out[0:4] = y[0:4]
    out[2] = 1
    struct.pack_into(order + "I", out, 8, t)
    c = 12
    d = 12
    f = 0
    a = [0] * 256
    i = 0
    s = 8
    s0 = 0
    h0 = 0
    h = 0
    while s < t:
        if i == 0:
            if d > e - 17:
                return bytes(y)
            i = 1
            out[c] = f
            c = d
            d += 1
            f = 0
        literal = s > t - 3
        if not literal:
            h = y[s] ^ y[s + 1]
            p = a[h]
            literal = p == 0 or y[s] != y[p]
        if s0 > 0:
            a[h0] = s0
            s0 = 0
        if literal:
            h0 = h
            s0 = s
            out[d] = y[s]
            d += 1
            s += 1
        else:
            a[h] = s
            f |= i
            p += 2
            s += 2
            r = s
            q = min(s + 255, t)
            while y[p] == y[s]:
                s += 1
                if s >= q:
                    break
                p += 1
            out[d] = h
            d += 1
            out[d] = s - r
            d += 1
        i = (i * 2) & 0xFF
    out[c] = f
    struct.pack_into(order + "I", out, 4, d)
    return bytes(out[:d])
