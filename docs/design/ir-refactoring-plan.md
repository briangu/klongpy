# KlongPy IR Refactoring Plan

**Status:** Draft
**Author:** Claude (with Brian Guarraci)
**Date:** 2026-02-04

## Overview

This document outlines a plan to introduce a proper Intermediate Representation (IR) to KlongPy, enabling:

1. **Source location tracking** - Error messages that point to exact positions in source code
2. **Linting** - Static analysis warnings before execution
3. **Better tooling** - IDE support, debuggers, profilers
4. **Optimization opportunities** - IR-level transformations

## Current Architecture

```
Source Text → kg_read/read_list → AST Nodes → eval() → Result
```

**Current AST Types** (in `klongpy/types.py` and `klongpy/parser.py`):
- `KGFn` - Function (body, args, arity)
- `KGCall` - Function invocation (extends KGFn)
- `KGOp` - Operator (op string, arity)
- `KGAdverb` - Adverb modifier
- `KGCond` - Conditional (extends list)
- `KGSym` - Symbol (extends str)
- `KGExprArray` - Evaluated array constructor (extends list)
- `KGChar` - Character (extends str)
- `KGLambda` - Python function wrapper

**Problems:**
- No source positions tracked anywhere
- Types extend builtins (str, list) making them hard to extend
- Mixed concerns: `KGFn` used for both AST and runtime values
- `eval()` directly interprets AST with no intermediate steps
- Error messages show Python tracebacks, not Klong source locations

## Proposed Architecture

```
Source Text → Lexer → Tokens → Parser → IR → [Linter] → Interpreter → Result
                 ↓                ↓
            (with positions)  (with spans)
```

## Phase 1: Source Positions in Tokens

**Goal:** Make the lexer emit tokens with source positions.

**Files to modify:**
- `klongpy/parser.py`

**New types:**

```python
@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A position in source code."""
    offset: int      # Byte offset from start
    line: int        # 1-indexed line number
    column: int      # 1-indexed column number

@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A range in source code."""
    start: SourceLocation
    end: SourceLocation
    source_id: str   # Identifies which source (file path, "<repl>", etc.)

@dataclass
class Token:
    """A lexical token with source position."""
    kind: str        # 'NUM', 'SYM', 'OP', 'STRING', 'LBRACKET', etc.
    value: Any       # The parsed value
    span: SourceSpan
```

**Changes to `kg_read`:**

```python
# Before
def kg_read(t, i, ...) -> tuple[int, Any]:
    ...
    return i, value

# After
def kg_read(t, i, source_id, ...) -> tuple[int, Token]:
    start_offset = i
    start_line, start_col = _offset_to_line_col(t, i)
    ...
    end_line, end_col = _offset_to_line_col(t, i)
    span = SourceSpan(
        SourceLocation(start_offset, start_line, start_col),
        SourceLocation(i, end_line, end_col),
        source_id
    )
    return i, Token(kind, value, span)
```

**Backward compatibility:**
- Add `legacy_mode` parameter defaulting to `True`
- When `True`, return old-style `(i, value)` tuples
- Gradually migrate callers

**Estimated scope:** ~200 lines changed in parser.py

## Phase 2: IR Node Types

**Goal:** Define proper IR types that carry source spans.

**New file:** `klongpy/ir.py`

```python
from dataclasses import dataclass
from typing import Any
from .parser import SourceSpan

@dataclass
class IRNode:
    """Base class for all IR nodes."""
    span: SourceSpan | None = None

@dataclass
class IRLiteral(IRNode):
    """A literal value (number, string, character)."""
    value: Any

@dataclass
class IRArray(IRNode):
    """An array literal [1 2 3]."""
    elements: list[Any]  # Already-evaluated values

@dataclass
class IRExprArray(IRNode):
    """An evaluated array [;expr1;expr2]."""
    exprs: list['IRNode']

@dataclass
class IRSym(IRNode):
    """A symbol reference."""
    name: str

@dataclass
class IROp(IRNode):
    """An operator."""
    op: str
    arity: int

@dataclass
class IRCall(IRNode):
    """A function/operator application."""
    fn: IRNode
    args: list[IRNode]
    arity: int

@dataclass
class IRFn(IRNode):
    """A function definition {body} or {[locals];body}."""
    body: list[IRNode]
    params: list[str] | None  # Local variable names
    arity: int

@dataclass
class IRCond(IRNode):
    """A conditional :[cond;then;else]."""
    condition: IRNode
    then_branch: IRNode
    else_branch: IRNode

@dataclass
class IRAdverb(IRNode):
    """An adverb-modified expression (f'x, f/x, etc.)."""
    adverb: str
    fn: IRNode
    args: list[IRNode]
    arity: int

@dataclass
class IRAssign(IRNode):
    """An assignment (::)."""
    target: IRSym
    value: IRNode

@dataclass
class IRProgram(IRNode):
    """A sequence of expressions."""
    exprs: list[IRNode]
```

**Estimated scope:** ~150 lines, new file

## Phase 3: Parser Produces IR

**Goal:** Modify parser to build IR nodes instead of current AST.

**Files to modify:**
- `klongpy/interpreter.py` - `_factor`, `_expr`, `prog`, `_read_fn_args`
- `klongpy/parser.py` - `read_list`, `read_cond`, `read_expr_array`

**Strategy:**
1. Create parallel `_factor_ir`, `_expr_ir`, `prog_ir` methods
2. Have them build `IRNode` trees
3. Keep old methods for backward compatibility initially
4. Add `use_ir=False` flag to `KlongInterpreter.__init__`

**Example transformation:**

```python
# Current _factor (simplified)
def _factor(self, t, i=0, ...):
    i, a = kg_read_array(t, i, ...)
    if safe_eq(a, '{'):
        i, a = self.prog(t, i, ...)
        a = KGFn(a, args=None, arity=get_fn_arity(a))
    return i, a

# New _factor_ir
def _factor_ir(self, t, i=0, source_id=None, ...):
    start = i
    i, token = kg_read(t, i, source_id, ...)

    if token.kind == 'LBRACE':
        i, body = self.prog_ir(t, i, source_id, ...)
        i = cexpect(t, i, '}')
        span = SourceSpan.merge(token.span, self._last_span)
        return i, IRFn(body, params=None, arity=..., span=span)

    if token.kind == 'SYM':
        return i, IRSym(token.value, span=token.span)

    # ... etc
```

**Estimated scope:** ~500 lines changed/added

## Phase 4: IR Interpreter

**Goal:** Interpreter that walks IR nodes with proper error handling.

**Files to modify:**
- `klongpy/interpreter.py` - new `eval_ir` method

```python
def eval_ir(self, node: IRNode):
    """Evaluate an IR node."""
    try:
        match node:
            case IRLiteral(value=v):
                return v

            case IRSym(name=n):
                try:
                    return self._context[KGSym(n)]
                except KeyError:
                    raise KlongError(f"undefined: {n}", node.span)

            case IRCall(fn=f, args=args, arity=arity):
                fn_val = self.eval_ir(f)
                arg_vals = [self.eval_ir(a) for a in args]
                return self._apply(fn_val, arg_vals, arity, node.span)

            case IRCond(condition=c, then_branch=t, else_branch=e):
                cond_val = self.eval_ir(c)
                if self._truthy(cond_val):
                    return self.eval_ir(t)
                else:
                    return self.eval_ir(e)

            case IRFn(body=body, params=params, arity=arity):
                # Return a callable that captures the IR
                return IRClosure(body, params, arity, self._context, node.span)

            case IRExprArray(exprs=exprs):
                results = [self.eval_ir(e) for e in exprs]
                return self._backend.kg_asarray(results)

            case IRProgram(exprs=exprs):
                result = None
                for expr in exprs:
                    result = self.eval_ir(expr)
                return result

            case _:
                raise KlongError(f"unknown IR node: {type(node)}", node.span)

    except KlongError:
        raise  # Already has span
    except Exception as e:
        raise KlongError(str(e), node.span) from e
```

**Estimated scope:** ~400 lines

## Phase 5: Enhanced Error Messages

**Goal:** Rich error messages with source context.

**New file:** `klongpy/errors.py`

```python
@dataclass
class KlongError(Exception):
    """An error with source location information."""
    message: str
    span: SourceSpan | None = None
    cause: 'KlongError | None' = None  # For error chains

    def format(self, source_registry: 'SourceRegistry') -> str:
        """Format error with source context."""
        lines = [f"Error: {self.message}"]

        if self.span:
            source = source_registry.get(self.span.source_id)
            if source:
                lines.append(f"  at {self.span.source_id}:{self.span.start.line}:{self.span.start.column}")
                lines.append(f"    {source.get_line(self.span.start.line)}")
                lines.append(f"    {' ' * (self.span.start.column - 1)}^")

        if self.cause:
            lines.append("")
            lines.append("Caused by:")
            lines.append(textwrap.indent(self.cause.format(source_registry), "  "))

        return '\n'.join(lines)


class SourceRegistry:
    """Registry of source texts for error reporting."""

    def __init__(self):
        self._sources: dict[str, str] = {}
        self._counter = 0

    def register(self, source: str, name: str | None = None) -> str:
        """Register source text, return source_id."""
        if name is None:
            name = f"<input-{self._counter}>"
            self._counter += 1
        self._sources[name] = source
        return name

    def get(self, source_id: str) -> 'SourceText | None':
        text = self._sources.get(source_id)
        return SourceText(text) if text else None


class SourceText:
    """Helper for extracting lines from source."""

    def __init__(self, text: str):
        self.text = text
        self._line_starts: list[int] | None = None

    def get_line(self, line_num: int) -> str:
        """Get a specific line (1-indexed)."""
        lines = self.text.splitlines()
        if 1 <= line_num <= len(lines):
            return lines[line_num - 1]
        return ""
```

**Example output:**

```
Error: undefined variable: foo
  at examples/test.kg:5:12
    result::bar(foo)
               ^^^

Caused by:
  Error: in function call
    at examples/test.kg:3:1
      bar::{x+foo}
      ^^^^^^^^^^^
```

**Estimated scope:** ~200 lines

## Phase 6: Linter

**Goal:** Static analysis that runs on IR before execution.

**New file:** `klongpy/linter.py`

```python
@dataclass
class LintMessage:
    level: str  # 'error', 'warning', 'info'
    message: str
    span: SourceSpan
    code: str   # e.g., 'W001', 'E001'

class Linter:
    def __init__(self):
        self.messages: list[LintMessage] = []

    def lint(self, node: IRNode, ctx: LintContext) -> None:
        """Recursively lint an IR tree."""
        self._check_node(node, ctx)
        for child in self._children(node):
            self.lint(child, ctx)

    def _check_node(self, node: IRNode, ctx: LintContext) -> None:
        match node:
            case IRSym(name=n, span=s):
                # W001: Symbol contains underscore
                if '_' in n and n not in ctx.known_syms:
                    self.warn('W001', s,
                        f"Symbol '{n}' contains underscore - "
                        f"did you mean the drop operator '_'?")

            case IRCall(fn=IRSym(name=n), span=s):
                # W002: Function used before definition
                if n not in ctx.defined and n not in BUILTINS:
                    self.warn('W002', s,
                        f"Function '{n}' used before definition")

            case IRFn(body=body, params=params, span=s):
                # W003: Unused local variable
                if params:
                    used = self._find_used_syms(body)
                    for p in params:
                        if p not in used and p not in ('x', 'y', 'z'):
                            self.warn('W003', s,
                                f"Local variable '{p}' declared but never used")

            case IRCall(fn=IROp(op='*'), args=[a, IRCall(fn=IROp(op='+'))], span=s):
                # W004: Precedence may be confusing
                self.info('W004', s,
                    "Expression 'a*b+c' evaluates as 'a*(b+c)' due to right-to-left evaluation")
```

**Lint checks to implement:**

| Code | Level | Description |
|------|-------|-------------|
| W001 | warning | Symbol contains underscore (likely operator confusion) |
| W002 | warning | Function/variable used before definition |
| W003 | warning | Local variable declared but unused |
| W004 | info | Potentially confusing operator precedence |
| W005 | warning | Unreachable code after return |
| W006 | warning | Empty function body |
| W007 | info | Function shadows builtin |
| E001 | error | Invalid syntax in array literal |
| E002 | error | Too many function parameters (>3) |

**Integration with REPL:**

```python
# In kgpy REPL
def execute(self, source: str):
    source_id = self.registry.register(source)
    ir = self.klong.parse_ir(source, source_id)

    # Run linter
    linter = Linter()
    linter.lint(ir, self.lint_context)
    for msg in linter.messages:
        self.print_lint_message(msg)

    if not linter.has_errors():
        return self.klong.eval_ir(ir)
```

**Estimated scope:** ~300 lines

## Phase 7: VSCode Integration

**Goal:** Language server protocol (LSP) support for the VSCode plugin.

This phase is more speculative but enabled by the IR:

1. **Diagnostics** - Send lint messages as diagnostics
2. **Hover** - Show type/value info for symbols
3. **Go to definition** - Jump to function definitions
4. **Find references** - Find all uses of a symbol
5. **Completion** - Suggest symbols in scope

**New package:** `klongpy-lsp` (separate repo)

Uses `pygls` or similar to implement LSP server that:
- Parses files to IR on change
- Runs linter, reports diagnostics
- Maintains symbol index for navigation

## Migration Strategy

### Backward Compatibility

1. All new functionality behind flags initially
2. Old API (`KGFn`, `KGOp`, etc.) remains working
3. Gradual deprecation warnings over 2-3 releases
4. Clear migration guide for users of Python API

### Testing Strategy

1. All existing tests must pass in both modes
2. New tests for IR-specific behavior
3. Property-based testing: `parse(source) |> eval` == `parse_ir(source) |> eval_ir`
4. Benchmark suite to ensure no performance regression

### Release Plan

| Version | Features |
|---------|----------|
| 0.8.0 | Phase 1-2: Token positions, IR types (internal) |
| 0.9.0 | Phase 3-4: IR parser/interpreter (opt-in via flag) |
| 0.10.0 | Phase 5: Enhanced error messages |
| 0.11.0 | Phase 6: Linter |
| 1.0.0 | IR mode default, deprecate legacy mode |
| 1.1.0 | Phase 7: LSP (separate package) |

## Open Questions

1. **Memory overhead** - How much does tracking spans cost? Benchmark needed.

2. **Serialization** - Should IR be serializable for caching/distribution?

3. **Optimization passes** - What IR transforms would be valuable?
   - Constant folding
   - Dead code elimination
   - Common subexpression elimination
   - Vectorization hints

4. **Debugging** - Should we support breakpoints/stepping at IR level?

5. **REPL experience** - How to handle incomplete expressions spanning multiple lines?

6. **Torch compilation** - Can we lower IR directly to torch.compile graphs?

## Appendix: Current Type Usage

Places where current AST types are used (would need updates):

```
klongpy/types.py      - KGFn, KGCall, KGOp, KGAdverb, KGCond, KGSym, KGChar
klongpy/parser.py     - kg_read, read_list, read_cond, KGExprArray
klongpy/interpreter.py - _factor, _expr, prog, eval, _eval_fn, call
klongpy/adverbs.py    - chain_adverbs, get_adverb_fn
klongpy/monads.py     - eval_monad_* functions
klongpy/dyads.py      - eval_dyad_* functions
klongpy/sys_fn.py     - Various system functions
klongpy/autograd.py   - Gradient computation
```

## References

- [Crafting Interpreters](https://craftinginterpreters.com/) - Nystrom's book on language implementation
- [Engineering a Compiler](https://www.cs.rice.edu/~keith/Errata.html) - Cooper & Torczon
- [pygls](https://github.com/openlawlibrary/pygls) - Python Language Server Protocol implementation
- [tree-sitter](https://tree-sitter.github.io/) - Alternative: generate parser from grammar
