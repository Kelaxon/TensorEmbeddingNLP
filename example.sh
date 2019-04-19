#!/usr/bin/env bash

label_portion=0.7
option='Pun'

win_size=5
cp_rank=10
neighbor=1000

seed=666

python load_data.py --option $option --label_portion $label_portion --seed $seed
python doc2vec.py --option $option --win_size $win_size --cp_rank $cp_rank
python label_propagation.py --option $option --neighbor $neighbor
# python fabp.py --option $option --neighbor $neighbor

