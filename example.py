#!/usr/bin/env python
__author__ = "kelaxon"
__github__ = "https://github.com/Kelaxon"

import subprocess
seed = 666
label_portion = 0.9
option = 'authorProfiling'

win_size = 5
cp_rank = 10
neighbor = 10

rebuild_data = True

subprocess.call(
    "python load_data.py --option {} --label_portion {} --seed {} --rebuild" \
        .format(option, label_portion, seed).split())
subprocess.run("python doc2vec.py --option {} --win_size {} --cp_rank {}" \
               .format(option, win_size, neighbor, cp_rank).split())
subprocess.run("python label_propagation.py --option {} --neighbor {}" \
               .format(option, win_size, neighbor).split())

