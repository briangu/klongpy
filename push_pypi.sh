#!/bin/sh

rm -rf dist
hatch build
twine upload dist/*
