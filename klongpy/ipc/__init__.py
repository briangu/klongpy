"""The kdb+ IPC wire protocol, written from public documentation.

Three layers; only the endpoint does I/O:

``klongpy.ipc.codec``     encode/decode one object (pure functions)
``klongpy.ipc.compress``  message compression and decompression
``klongpy.ipc.session``   handshake, framing and response matching (state machine)
``klongpy.ipc.endpoint``  requests, serving, calls, errors on the wire, handshake glue
                          (used by ``klongpy.sys_fn_ipc``; imported from there, not here)
``klongpy.ipc.legacy``    the deprecated pickle protocol, kept for one transition release

This is an independent implementation of the protocol as described by public
documentation and Apache-licensed client code; kdb+ is a trademark of KX.
"""
from .errors import IPCError, ProtocolError, IPCTypeError, IPCRangeError, IPCStateError
from .codec import (
    encode, encode_call, decode, KdbLambda, KdbOpaque, KdbError, set_function_text_hook, set_table_factory, KdbTable,
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
    "set_function_text_hook", "set_table_factory", "KdbTable", "MAX_DEPTH", "compress", "decompress", "MAX_MESSAGE",
    "Session", "Pending", "Service", "Handshake", "HandshakeAccepted", "HandshakeRejected",
    "Message", "Failure", "Closed", "NestingLimit",
]
