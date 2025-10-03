#!/bin/bash
set -e

uv init --no-readme --no-pin-python
uv add "eval_recipes @ git+https://github.com/microsoft/eval-recipes@v0.0.6"
