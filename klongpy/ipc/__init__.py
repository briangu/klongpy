"""The kdb+ IPC wire protocol, written from public documentation (ticket 105).

Three layers, none of which does I/O except the endpoint (not here yet):

``klongpy.ipc.codec``     encode/decode one object (pure functions)
``klongpy.ipc.compress``  message compression and decompression
``klongpy.ipc.session``   handshake, framing and response matching (state machine)

This is an independent implementation of the protocol as described by public
documentation and Apache-licensed client code; kdb+ is a trademark of KX.
"""
from .errors import IPCError, ProtocolError, IPCTypeError, IPCRangeError, IPCStateError
from .codec import (
    encode, encode_call, decode, KdbLambda, KdbOpaque, KdbError, set_function_text_hook,
    MAX_DEPTH,
)
from .compress import compress, decompress, MAX_MESSAGE
from .session import (
    Session, Pending, Service, Handshake, HandshakeAccepted, HandshakeRejected,
    Message, Failure, Closed, NestingLimit,
)

__all__ = [
    "IPCError", "ProtocolError", "IPCTypeError", "IPCRangeError", "IPCStateError",
    "encode", "encode_call", "decode", "KdbLambda", "KdbOpaque", "KdbError",
    "set_function_text_hook", "MAX_DEPTH", "compress", "decompress", "MAX_MESSAGE",
    "Session", "Pending", "Service", "Handshake", "HandshakeAccepted", "HandshakeRejected",
    "Message", "Failure", "Closed", "NestingLimit",
]
