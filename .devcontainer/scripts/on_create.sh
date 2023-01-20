#!/bin/bash

pip install --upgrade PIP
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install --install-hooks
