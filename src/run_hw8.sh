#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /dropbox/21-22/575k/env
python run.py --train_source /dropbox/21-22/575k/data/europarl-v7-es-en/train.en.txt --train_target /dropbox/21-22/575k/data/europarl-v7-es-en/train.es.txt --output_file test.en.txt.es --num epochs 8 --embedding dim 16 --hidden dim 64 --num layers 2
