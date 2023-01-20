#!/bin/bash

pip install --upgrade PIP
pip install -r requirements.txt -r requirements-dev.txt -r requirements-m1.txt
pre-commit install --install-hooks
