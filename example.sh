#!/usr/bin/env bash

label_portion=0.7
option='16000oneliners'

win_size=30
cp_rank=20
neighbor=20

seed=345

# python load_data.py --option $option --label_portion $label_portion --seed $seed
# python doc2vec.py --option $option --win_size $win_size --cp_rank $cp_rank
python label_propagation.py --option $option --neighbor $neighbor
# python fabp.py --option $option --neighbor $neighbor

