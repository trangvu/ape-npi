#!/usr/bin/env python3

import argparse
import sys
import os
import re
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)
tercom_path = os.path.join(script_dir, 'tercom.jar')

from translate.evaluation import corpus_bleu, corpus_ter, corpus_wer, corpus_cer, corpus_bleu1

parser = argparse.ArgumentParser()
parser.add_argument('mt', nargs='+')
parser.add_argument('ref')

parser.add_argument('--src')
parser.add_argument('--min', type=int, default=0)
parser.add_argument('--max', type=int, default=70)
parser.add_argument('--step', type=int, default=5)
parser.add_argument('--labels', nargs='*')
parser.add_argument('--output')

parser.add_argument('--bar', action='store_true')

args = parser.parse_args()

if args.src is None:
    args.src = args.ref

assert args.labels is None or len(args.labels) == len(args.mt)

for k, mt in enumerate(args.mt):
    with open(args.src) as src_file, open(mt) as mt_file, open(args.ref) as ref_file:
        lines = list(zip(src_file, mt_file, ref_file))

        bins = OrderedDict()

        for i in range(args.min, args.max, args.step):
            lines_ = [(mt.strip(), ref.strip()) for src, mt, ref in lines if i < len(src.split()) <= i + args.step]
            if len(lines_) > 0:
                score, summary = corpus_bleu(*zip(*lines_))
                bins[i + args.step] = score
                # print(i + args.step, '{:.1f}'.format(score), len(lines_), summary)

        values = np.array(list(bins.values()))
        keys = np.array(list(bins.keys()))

        label = args.labels[k] if args.labels else None

        if args.bar:
            width = 1 if len(args.mt) > 1 else args.step - 1
            keys += k
            plt.bar(keys + k, values, width=width, label=label)
        else:
            plt.plot(keys, values, label=label)

xlabel = 'Reference words' if args.src == args.ref else 'Source words'
plt.xlabel(xlabel)
plt.ylabel('BLEU')
plt.legend()

if args.output:
    plt.savefig(args.output)
else:
    plt.show()
