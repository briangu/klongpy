import sys
from .core import (
    KGLambda,
    KGCall,
    KGSym,
    KlongException,
    reserved_fn_args,
    reserved_fn_symbol_map,
)

try:
    from qpython import qconnection
except Exception:  # ImportError or other
    qconnection = None

class KdbConnection:
    def __init__(self, host='localhost', port=5000, user=None, password=None):
        if qconnection is None:
            raise KlongException("qpython not installed")
        self.conn = qconnection.QConnection(host=host, port=port, username=user, password=password)
        self.conn.open()

    def execute(self, query):
        return self.conn(query)

    def close(self):
        if self.conn.is_connected():
            self.conn.close()

    def is_open(self):
        return self.conn.is_connected()

    def __str__(self):
        return f"q[{self.conn.host}:{self.conn.port}]"

class KdbFnProxy(KGLambda):
    def __init__(self, connection, name, arity=1):
        self.connection = connection
        self.name = name
        self.args = reserved_fn_args[:arity]

    def __call__(self, _, ctx):
        params = [ctx[reserved_fn_symbol_map[x]] for x in self.args]
        args_str = ','.join(map(str, params))
        qexp = f"{self.name} {args_str}" if args_str else self.name
        return self.connection.execute(qexp)

    def __str__(self):
        return f"{str(self.connection)}:{self.name}:fn"

class KdbDictHandle(dict):
    def __init__(self, connection):
        self.connection = connection

    def __getitem__(self, x):
        return self.get(x)

    def __setitem__(self, x, y):
        return self.set(x, y)

    def __contains__(self, x):
        try:
            _ = self.get(x)
            return True
        except Exception:
            return False

    def get(self, x):
        if isinstance(x, KGSym):
            qexp = f"value {x}"
            r = self.connection.execute(qexp)
            if isinstance(r, qconnection.QFunction):
                return KdbFnProxy(self.connection, str(x))
            return r
        else:
            return self.connection.execute(x)

    def set(self, x, y):
        if isinstance(x, KGSym):
            qexp = f"{x}::{y}"
        else:
            qexp = str(x)
        self.connection.execute(qexp)
        return self

    def close(self):
        self.connection.close()

    def is_open(self):
        return self.connection.is_open()

    def __str__(self):
        return f"{str(self.connection)}:dict"

# evaluation functions

def eval_sys_fn_qcli(klong, x):
    """
        .qcli(x)                                 [Create-KDB-client]

        Create a kdb+ IPC client. "x" may be an integer interpreted as a port on localhost, or a string "<host>:<port>".
        Returns a remote function handle which executes q expressions on the server.
    """
    x = x.a if isinstance(x, KGCall) else x
    if isinstance(x, KdbConnection):
        conn = x
    elif isinstance(x, KdbDictHandle):
        conn = x.connection
    else:
        addr = str(x)
        parts = addr.split(":")
        host = parts[0] if len(parts) > 1 else "localhost"
        port = int(parts[0] if len(parts) == 1 else parts[1])
        conn = KdbConnection(host=host, port=port)
    return KdbFnProxy(conn, "")


def eval_sys_fn_qclid(klong, x):
    """
        .qclid(x)                            [Create-KDB-dict-client]

        Similar to .qcli but returns a dictionary style handle for remote access.
    """
    x = x.a if isinstance(x, KGCall) else x
    if isinstance(x, KdbDictHandle):
        return x
    if isinstance(x, KdbConnection):
        conn = x
    else:
        addr = str(x)
        parts = addr.split(":")
        host = parts[0] if len(parts) > 1 else "localhost"
        port = int(parts[0] if len(parts) == 1 else parts[1])
        conn = KdbConnection(host=host, port=port)
    return KdbDictHandle(conn)


def eval_sys_fn_qclic(x):
    """
        .qclic(x)                                 [Close-KDB-client]

        Close a kdb+ IPC connection opened via .qcli or .qclid.
    """
    if isinstance(x, KGCall):
        x = x.a
    if isinstance(x, (KdbDictHandle, KdbConnection, KdbFnProxy)):
        conn = x.connection if hasattr(x, 'connection') else x
        if conn.is_open():
            conn.close()
            return 1
    return 0


def create_system_functions_kdb():
    def _get_name(s):
        i = s.index(".")
        return s[i : i + s[i:].index("(")]

    registry = {}
    m = sys.modules[__name__]
    for x in filter(lambda n: n.startswith("eval_sys_"), dir(m)):
        fn = getattr(m, x)
        registry[_get_name(fn.__doc__)] = fn
    return registry
