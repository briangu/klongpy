"""
Regression tests for the Not monad (~) result dtype.

History: ~ used to force ``dtype=object`` on its result (via
``bknp.asarray(x, dtype=object)`` in eval_monad_not). For numeric input that
made ~ return an *object* array, which then could not be used as an index or by
``&`` (where) — ``&~v`` / ``RR@(&~v)`` failed with::

    Cannot cast array data from dtype('O') to dtype('int64')

This blocked the very common Klong idiom of masking with a negated predicate
(there is no >= operator, so ``~(a<b)`` is how you express it). The fix keeps a
clean integer/bool result for numeric/boolean input while preserving the old
object-based Python-``not`` semantics for strings/chars.

Run on both backends:
    pytest tests/test_not_dtype.py -v
    pytest tests/test_not_dtype.py -v --backend torch --device cpu
"""
import numpy as np


def _np(x):
    """Materialize a Klong result (numpy array or torch tensor) as a numpy array."""
    try:
        x = x.cpu().numpy()
    except AttributeError:
        pass
    return np.asarray(x)


class TestNotDtype:
    def test_not_of_numeric_array_is_integer_dtype(self, klong):
        # ~ of a numeric/boolean array must NOT be object dtype, otherwise it is
        # unusable as an index / by & (where).
        r = _np(klong('~[1 0 0 1]'))
        assert r.dtype != object
        assert r.dtype.kind in ('b', 'i', 'u')
        assert r.tolist() == [0, 1, 1, 0]

    def test_not_of_comparison_is_integer_dtype(self, klong):
        klong('CNT::[10.0 20.0 25.0 5.0]')
        klong('MINOBS::20')
        r = _np(klong('~CNT<MINOBS'))
        assert r.dtype != object
        assert r.tolist() == [0, 1, 1, 0]

    def test_where_of_negated_predicate(self, klong):
        # & (where) over a negated predicate — the original failure mode.
        klong('CNT::[10.0 20.0 25.0 5.0]')
        klong('MINOBS::20')
        r = _np(klong('&~CNT<MINOBS'))
        assert r.tolist() == [1, 2]

    def test_index_by_negated_predicate(self, klong):
        # Index a matrix's rows by &~predicate — the downstream allocator idiom.
        klong('CNT::[10.0 20.0 25.0 5.0]')
        klong('MINOBS::20')
        klong('RR::[[0 1 2] [3 4 5] [6 7 8] [9 10 11]]')
        r = _np(klong('RR@(&~CNT<MINOBS)'))
        assert r.tolist() == [[3, 4, 5], [6, 7, 8]]

    def test_not_preserves_scalar_semantics(self, klong):
        assert int(_np(klong('~0'))) == 1
        assert int(_np(klong('~1'))) == 0
        assert int(_np(klong('~123'))) == 0

    def test_not_preserves_string_semantics(self, klong):
        # Non-numeric (string) input keeps Python-`not` semantics.
        assert int(_np(klong('~""'))) == 1
        assert int(_np(klong('~"string"'))) == 0

    def test_not_of_empty_and_symbol(self, klong):
        assert int(_np(klong('~[]'))) == 1
        assert int(_np(klong('~:foo'))) == 0

    def test_not_of_nested_object_array(self, klong):
        # Heterogeneous (ragged) input still negates element-wise.
        klong('NEST::[[1 0] [0 0 1]]')
        r = klong('~NEST')
        assert [_np(z).tolist() for z in r] == [[0, 1], [1, 1, 0]]
