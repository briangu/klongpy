"""Exceptions of the kdb+ IPC layers (codec, compression, session).

The decoder and the decompressor raise :class:`ProtocolError` on any malformed
input and nothing else (except :class:`IPCRangeError` when a 32-bit build is
asked to narrow a value that does not fit).  The endpoint layer maps these to
``KlongError`` kinds later (ticket 105 step 3, ticket 040).
"""


class IPCError(Exception):
    """Base class of every error raised by klongpy.ipc."""

    #: wire name of the Klong error kind this maps to (see design section 7)
    kind = "system"


class ProtocolError(IPCError):
    """The peer sent bytes that are not a valid kdb+ IPC message (kind ``system``).

    Always fatal for the connection.
    """

    kind = "system"


class IPCTypeError(IPCError):
    """A Klong value that has no kdb+ representation (kind ``type``)."""

    kind = "type"


class IPCRangeError(IPCError):
    """A number that does not fit the target width (kind ``range``)."""

    kind = "range"


class IPCStateError(RuntimeError):
    """The session API was used out of order (a programming error, not a peer error)."""
