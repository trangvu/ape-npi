#!/usr/bin/env bash

root_dir=`pwd`

/usr/bin/env pip3 install tensorflow-gpu python-dateutil pyyaml matplotlib --user --upgrade

cat >>~/.bashrc << EOL
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/cuda/lib64
alias get-best-score=${root_dir}/scripts/get-best-score.py
alias plot-loss=${root_dir}/scripts/plot-loss.py
alias multi-print=${root_dir}/scripts/multi-print.py
alias copy-model=${root_dir}/scripts/copy-model.py
EOL
