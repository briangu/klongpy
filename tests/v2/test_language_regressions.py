"""Interactions among projections, module lifetimes, and multi-name binding."""

import pytest

from .runtime import PROGRAMS, evaluate


@pytest.mark.parametrize("setup", [
    'f::counter.make(10)',
    'f::counter.make(10);counter::0',
], ids=["module-bound", "only-projection-remains"])
def test_projection_keeps_module_state(host, setup):
    expr = f'.l("./counter");{setup};first::f(2);[;first;f(2)]'
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == "[13 14]"


def test_projection_from_locally_loaded_module(host):
    expr = ('openCounter()::{[counter];.l("./counter");counter.make(10)};'
            'f::openCounter();f(2)')
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == "13"


def test_module_reloads_after_last_projection_is_released(host):
    expr = ('.l("./counter");f::counter.make(10);first::f(2);counter::0;f::0;'
            '.l("./counter");g::counter.make(10);[;first;g(2)]')
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == "[13 13]"


def test_failed_destructure_preserves_existing_binding(host, tmp_path):
    # After the first statement, destructuring follows ordinary assignment rules.
    expr = ('a::42;f()::{0;[a b]::[8 9]};r::.try(f;[]);'
            '[;r@0;(r@1)@"kind";a]')
    assert evaluate(host, expr, tmp_path, options=["--strict", "2"]) == '[0 "name" 42]'


def test_failed_destructure_restores_local_and_aliased_fields(host, tmp_path):
    expr = ('d:::{;"old";4};alias::d;f()::{[a r];a::42;'
            'r::.try({0;[a d.old d.new missing]::[8 9 10 11]};[]);'
            '[;r@0;a;alias.old;:_alias@"new"]};f()')
    assert evaluate(host, expr, tmp_path, options=["--strict", "2"]) == "[0 42 4 1]"


def test_destructure_can_rebind_a_dictionary_before_writing_its_field(host, tmp_path):
    expr = 'd::0;[d d.k]::[;:{};7];d.k'
    assert evaluate(host, expr, tmp_path, options=["--strict", "2"]) == "7"


def test_failed_selective_import_checks_caller_scope(host):
    # Both exports exist and are valid names. Only the first exists in caller scope.
    expr = ('f()::{[ok r];ok::42;r::.try({.l("./counter";["ok" "make"])};[]);'
            '[;r@0;(r@1)@"kind";ok]};f()')
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == '[0 "name" 42]'


@pytest.mark.parametrize("key", ["bad-key", "absent"], ids=["invalid-binding-name", "missing-export"])
def test_failed_selective_import_preserves_existing_binding(host, key):
    expr = (f'ok::42;r::.try({{.l("./counter";["ok" "{key}"])}};[]);'
            '[;r@0;(r@1)@"kind";ok]')
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == '[0 "name" 42]'


def test_selective_import_binds_all_valid_names(host):
    expr = '.l("./counter";["make" "ok"]);f::make(10);[;ok;f(2)]'
    assert evaluate(host, expr, PROGRAMS, options=["--strict", "2"]) == "[7 13]"


def test_shared_assertion_returns_success(host, tmp_path):
    expr = '.l("checks");checks.require(1;"range";"expected positive width")'
    assert evaluate(host, expr, tmp_path, options=["--strict", "2"]) == "1"


@pytest.mark.parametrize("kind", ["range", "system"])
def test_shared_assertion_preserves_error_kind_and_message(host, tmp_path, kind):
    expr = (f'.l("checks");r::.try({{checks.require(0;"{kind}";"bad batch")}};[]);'
            '[;r@0;(r@1)@"kind";(r@1)@"message"]')
    assert evaluate(host, expr, tmp_path, options=["--strict", "2"]) == f'[0 "{kind}" "bad batch"]'


@pytest.mark.parametrize("record", ['[;x;"ok"]', '[;x;helper]'], ids=["plain-records", "function-records"])
def test_module_sweep_defers_when_workspace_is_full(kg, tmp_path, record):
    (tmp_path / "mix.kg").write_text(
        f"helper(a)::{{a}}\nitems::{{{record}}}'!1000\n"
        'size()::{#items}\nexport:::{;"size";size}\n'
    )
    expr = '.l("./mix");answer::mix.size();mix::0;answer'
    assert evaluate(kg, expr, tmp_path, options=["--heap-limit", "160K"]) == "1000"
