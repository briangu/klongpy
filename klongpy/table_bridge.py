"""Optional Pandas and Arrow representations of native Klong tables."""

from collections.abc import ItemsView, KeysView, ValuesView
import sys

import numpy as np

from .ipc import KdbTable
from .types import KGSym, KLONG_UNDEFINED


class _WireTableView(KdbTable):
    """Keep the Python mapping interface of an unkeyed table read from q."""

    def __init__(self, table):
        super().__init__()
        self._native_table = table

    @property
    def columns(self):
        return self._native_table.columns

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        return iter(map(KGSym, self.columns))

    def __contains__(self, name):
        return isinstance(name, str) and str(name) in self.columns

    def __getitem__(self, name):
        if name not in self:
            raise KeyError(name)
        return self._native_table[name]

    def __setitem__(self, name, values):
        self._native_table[name] = values

    def get(self, name, default=None):
        return self[name] if name in self else default

    def keys(self):
        return KeysView(self)

    def values(self):
        return ValuesView(self)

    def items(self):
        return _WireItems(self)

    def get_dataframe(self):
        return self._native_table.get_dataframe()

    def __arrow_c_stream__(self, requested_schema=None):
        return self._native_table.__arrow_c_stream__(requested_schema)

    def __klong_ipc_body__(self):
        return self._native_table.__klong_ipc_body__()

    def __repr__(self):
        return repr(dict(self.items()))


class _WireItems(ItemsView):
    def __iter__(self):
        # Arrow treats a dict subclass as a column mapping. Pass the logical
        # type on each view for this path, including empty and nullable columns.
        for name in self._mapping:
            values = self._mapping[name]
            qtype = self._mapping._native_table._ipc_column_type(name)
            yield name, _WireColumn(values, qtype) if qtype else values


class _WireColumn(np.ndarray):
    def __new__(cls, values, qtype):
        col = np.asarray(values).view(cls)
        col.qtype = qtype
        return col

    def __array_finalize__(self, original):
        self.qtype = getattr(original, "qtype", None)

    def __arrow_array__(self, type=None):
        pa = sys.modules["pyarrow"]
        values = [None if value is KLONG_UNDEFINED else value for value in self.tolist()]
        col = _q_arrow_column(pa, values, self.qtype)
        return col if type is None else col.cast(type)


_NAT = -(1 << 63)
_UNIX_TO_Q_DAYS = 10957
_NANOSECONDS_PER_DAY = 86_400_000_000_000


def _arrow_integer_type(pa, dtype):
    types = pa.types
    if types.is_date32(dtype) or types.is_time32(dtype):
        return pa.int32()
    if any(check(dtype) for check in (types.is_timestamp, types.is_duration, types.is_time64, types.is_date64)):
        return pa.int64()
    return None


def _pandas_temporal_dtype(pd, dtype):
    if isinstance(dtype, pd.DatetimeTZDtype):
        return np.dtype(f"datetime64[{dtype.unit}]")
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return dtype
    return None


def _pandas_values(pd, series):
    if _pandas_temporal_dtype(pd, series.dtype) is not None:
        return [None if v == _NAT else v for v in series.array.asi8.tolist()]
    return [None if v is pd.NA or v is pd.NaT else v for v in series.tolist()]


def import_table(source):
    """Convert table inputs to ordered columns of native scalar values and an export schema."""
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(source, pd.DataFrame):
        columns = [(name, _pandas_values(pd, source.iloc[:, i])) for i, name in enumerate(source.columns)]
        return columns, ("pandas", dict(source.dtypes.items()))

    pa = sys.modules.get("pyarrow")
    if pa is not None and isinstance(source, (pa.Table, pa.RecordBatch)):
        columns = []
        for i, field in enumerate(source.schema):
            col = source.column(i)
            physical = _arrow_integer_type(pa, field.type)
            columns.append((field.name, (col if physical is None else col.cast(physical)).to_pylist()))
        return columns, ("arrow", source.schema)
    return None


def _q_column(values, qtype):
    """Translate q's temporal epoch and units to a NumPy or Arrow-compatible dtype."""
    plain = {1: "bool", 4: "uint8", 5: "int16", 6: "int32", 7: "int64", 8: "float32", 9: "float64",
             10: "string", 11: "string"}
    if qtype in plain:
        return values, plain[qtype]
    if qtype == 12:
        return [None if v is None else v + _UNIX_TO_Q_DAYS * _NANOSECONDS_PER_DAY for v in values], "datetime64[ns]"
    if qtype == 13:
        origin = np.datetime64("2000-01", "M")
        epoch_days = lambda months: int((origin + np.timedelta64(months, "M")).astype("datetime64[D]").astype("int64"))
        return [None if v is None else epoch_days(v) for v in values], "datetime64[D]"
    if qtype == 14:
        return [None if v is None else v + _UNIX_TO_Q_DAYS for v in values], "datetime64[D]"
    if qtype == 15:
        return [None if v is None else round(v * 86_400_000) + _UNIX_TO_Q_DAYS * 86_400_000 for v in values], "datetime64[ms]"
    if qtype == 16:
        return values, "timedelta64[ns]"
    if qtype == 17:
        return [None if v is None else v * 60 for v in values], "timedelta64[s]"
    if qtype == 18:
        return values, "timedelta64[s]"
    if qtype == 19:
        return values, "timedelta64[ms]"
    return values, None


def _pandas_array(pd, values, dtype):
    temporal = _pandas_temporal_dtype(pd, dtype)
    if temporal is None:
        return pd.array(values, dtype=dtype)
    physical = np.array([_NAT if value is None else value for value in values], dtype="int64")
    result = physical.view(temporal)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return pd.DatetimeIndex(result, tz="UTC").tz_convert(dtype.tz).array
    return pd.array(result, dtype=dtype)


def export_dataframe(pd, columns, schema):
    origin, dtypes = schema if schema is not None else (None, None)
    if origin == "arrow":
        return export_arrow(sys.modules["pyarrow"], columns, schema).to_pandas(types_mapper=pd.ArrowDtype)
    out = {}
    for name, values, qtype in columns:
        if origin == "pandas" and name in dtypes:
            out[name] = _pandas_array(pd, values, dtypes[name])
            continue
        values, dtype = _q_column(values, qtype)
        if dtype is not None and (dtype.startswith("int") or dtype.startswith("uint")) and None in values:
            dtype = dtype.replace("int", "Int").replace("u", "U")
        if dtype == "datetime64[D]":
            values = [None if v is None else v * 86_400 for v in values]
            dtype = "datetime64[s]"
        if dtype == "bool" and None in values:
            dtype = "boolean"
        out[name] = pd.array(values) if dtype is None else _pandas_array(pd, values, dtype)
    return pd.DataFrame(out)


def _arrow_array(pa, values, dtype=None):
    if dtype is not None and pa.types.is_boolean(dtype):
        values = [None if v is None else bool(v) for v in values]
    return pa.array(values, type=dtype)


def _q_arrow_column(pa, values, qtype):
    values, dtype = _q_column(values, qtype)
    if dtype == "datetime64[D]":
        arrow_type = pa.date32()
    elif dtype == "string":
        arrow_type = pa.string()
    else:
        arrow_type = pa.from_numpy_dtype(dtype) if dtype is not None else None
    return _arrow_array(pa, values, arrow_type)


def export_arrow(pa, columns, schema):
    origin, dtypes = schema if schema is not None else (None, None)
    if origin == "pandas":
        return pa.Table.from_pandas(export_dataframe(sys.modules["pandas"], columns, schema), preserve_index=False)
    fields = []
    arrays = []
    for name, values, qtype in columns:
        if origin == "arrow" and name in dtypes.names:
            field = dtypes.field(name)
            col = _arrow_array(pa, values, field.type)
        else:
            col = _q_arrow_column(pa, values, qtype)
            field = pa.field(name, col.type)
        fields.append(field)
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=dtypes.metadata if origin == "arrow" else None))
