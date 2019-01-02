#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import re
from translate.evaluation import corpus_bleu, corpus_ter, corpus_wer, tercom_statistics
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--bleu', action='store_true')
#parser.add_argument('--ter', action='store_true')
#parser.add_argument('--wer', action='store_true')
#parser.add_argument('--all', '-a', action='store_true')
parser.add_argument('--max-size', type=int)
parser.add_argument('--case-insensitive', '-i', action='store_true')

parser.add_argument('--draws', type=int, default=1000)
parser.add_argument('--sample-size', type=int, default=0)
parser.add_argument('-p', type=float, default=0.05)


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.source) as src_file, open(args.target) as trg_file:
        if args.case_insensitive:
            hypotheses = [line.strip().lower() for line in src_file]
            references = [line.strip().lower() for line in trg_file]
        else:
            hypotheses = [line.strip() for line in src_file]
            references = [line.strip() for line in trg_file]

        if args.max_size is not None:
            hypotheses = hypotheses[:args.max_size]
            references = references[:args.max_size]

        if len(hypotheses) != len(references):
            sys.stderr.write('warning: source and target don\'t have the same length\n')
            size = min(len(hypotheses), len(references))
            hypotheses = hypotheses[:size]
            references = references[:size]

        avg_stats, stats = tercom_statistics(hypotheses, references)

        ters = [stats_['TER'] for stats_ in stats]

        mean = sum(ters) / len(ters)
        variance = sum((ter - mean) ** 2 for ter in ters) / (len(ters) - 1)

        ts = {0.01: 2.5841, 0.05: 1.9639, 0.10: 1.6474}
        t = ts.get(args.p)
        if t is None:
            raise Exception

        d = t * np.sqrt(variance / len(ters))

        print('{:.3f} +/- {:.3f}'.format(mean, d))