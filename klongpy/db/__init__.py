from .sys_fn_db import create_system_functions_db
from .sys_fn_kvs import create_system_functions_kvs

db_exports = create_system_functions_db()
kvs_exports = create_system_functions_kvs()

klong_exports = {}
klong_exports.update(db_exports)
klong_exports.update(kvs_exports)
