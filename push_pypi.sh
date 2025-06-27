#!/bin/bash -e
rm -rf dist
python -m pip install --upgrade build cibuildwheel twine
python -m build --sdist -o dist
python -m cibuildwheel --output-dir dist
twine upload dist/*

