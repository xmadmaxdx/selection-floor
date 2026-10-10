#!/usr/bin/env bash
set -e
# Colab / fresh-VM setup. Run with: !bash colab_setup.sh
python --version
pip install -q -r requirements.txt
echo "setup ok"