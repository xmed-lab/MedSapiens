#!/bin/bash

## Function to install a package via pip with editable mode and verbose output
pip_install_editable() {
  print_green "Installing $1..."
  cd "$1" || exit
  pip install -e . -v
  cd - || exit
  print_green "Finished installing $1."
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

