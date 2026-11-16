"""Exceptions of the kdb+ IPC layers (codec, compression, session).

The decoder and the decompressor raise :class:`ProtocolError` on any malformed
input and nothing else (except :class:`IPCRangeError` when a 32-bit build is
asked to narrow a value that does not fit).  The endpoint layer maps these to
``KlongError`` classes by their ``kind``.  The ``KlongIPC*`` classes at the end
are what Klong programs see when the transport fails.
"""
from ..errors import KlongSystemError


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


# ----------------------------------------------------------------------------
# transport failures seen by Klong programs (raised by the endpoint layer)
# ----------------------------------------------------------------------------

class KlongIPCException(KlongSystemError):
    """The connection failed or could not be made (kind ``system``)."""


class KlongIPCConnectionClosedException(KlongIPCException):
    pass


class KlongIPCConnectionFailureException(KlongIPCException):
    pass


class KlongIPCTimeoutException(KlongIPCConnectionFailureException, TimeoutError):
    """A call got no answer in time (``call_timeout``); the connection was closed."""


class KlongIPCCreateConnectionException(KlongIPCException):
    pass


class KlongIPCHandshakeException(KlongIPCCreateConnectionException):
    """The kdb+ handshake failed: refused, timed out or malformed."""
