from .sys_fn_db import create_system_functions_db
from .sys_fn_kvs import create_system_functions_kvs

db_exports = create_system_functions_db()
kvs_exports = create_system_functions_kvs()

klongpy_exports = {}
klongpy_exports.update(db_exports)
klongpy_exports.update(kvs_exports)
