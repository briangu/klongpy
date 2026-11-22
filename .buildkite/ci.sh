#!/bin/sh
# One CI stage per call: `ci.sh engine`, `ci.sh python`, `ci.sh docs`. Runs in a clean Linux container that has Python and
# a C toolchain; it installs the Rust toolchain and the Python packages it needs, so the same command works on a laptop:
#   docker run --rm -v "$PWD":/src:ro python:3.12-bookworm sh -c 'cp -r /src /work && cd /work && .buildkite/ci.sh engine'
set -eu

stage="${1:?usage: ci.sh engine|python|docs}"
cd "$(dirname "$0")/.."

if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
. "$HOME/.cargo/env" 2>/dev/null || true
export SETUPTOOLS_RUST_CARGO_PROFILE=release

install_package() {
  python -m venv /tmp/ci-venv
  . /tmp/ci-venv/bin/activate
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e ".[web,db,ws,repl,dev]"
}

case "$stage" in
  engine)
    # the interpreter alone: its own tests, then the language conformance cases through the kg binary
    (cd native && cargo build --release && cargo test --release --workspace --no-fail-fast)
    python -m venv /tmp/ci-venv
    . /tmp/ci-venv/bin/activate
    python conformance/run.py --cmd "native/target/release/kg --conformance" --jobs 4
    ;;
  python)
    # the Python package on the engine: conformance in process, then the test suite on NumPy
    install_package
    python conformance/run.py
    python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_compiler.py
    ;;
  docs)
    # every example in the documentation runs and gives what the page shows; the site builds without warnings
    install_package
    (cd native && cargo build --release -p klong-cli)
    python -m pip install --quiet mkdocs mkdocs-material
    python scripts/check_docs.py
    python -m mkdocs build --strict
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac
