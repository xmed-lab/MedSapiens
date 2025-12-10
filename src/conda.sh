#!/bin/bash

## Function to install a package via pip with editable mode and verbose output
pip_install_editable() {
  cd "$1" || exit
  pip install -e . -v
  cd - || exit
}

# Install engine
pip_install_editable "engine"
#
pip_install_editable "pretrain"
#
## Install pose
pip_install_editable "pose"
#
## Install det
pip_install_editable "det"
#

