#!/bin/bash

set -e

. ${HOME}/miniconda/bin/activate
export LD_LIBRARY_PATH="${CONDA_PATH}/lib:${LD_LIBRARY_PATH}"

# Uninstall previous versions of PyTorch. Doing this twice is intentional.
# Error messages about torch not being installed are benign.
pip uninstall -y torch || true
pip uninstall -y torch || true

pip install -r requirements.txt

pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
