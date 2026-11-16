#!/bin/bash -e
# Build and upload a release. Requires: pip install "klongpy[release]"
cd "$(dirname "$0")"
rm -rf dist
python3 -m build
twine check dist/*
twine upload dist/*
