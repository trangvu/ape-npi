#!/usr/bin/env python3

import argparse
import sys
import numpy as np
import re
from translate.evaluation import corpus_bleu, corpus_ter, corpus_wer
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

        indices = np.arange(len(hypotheses))
        if args.sample_size == 0:
            args.sample_size = len(hypotheses)

        bleu_scores = []
        hypotheses = np.array(hypotheses)
        references = np.array(references)

        for _ in range(args.draws):
            indices = np.random.randint(len(hypotheses), size=args.sample_size)
            hypotheses_ = hypotheses[indices]
            references_ = references[indices]

            bleu, _ = corpus_bleu(hypotheses_, references_)
            bleu_scores.append(bleu)

        bleu_scores = sorted(bleu_scores)
        k = int(len(bleu_scores) * args.p) // 2   # FIXME

        bleu_scores = bleu_scores[k:len(bleu_scores) - k]

        print('[{:.3f}, {:.3f}]'.format(bleu_scores[0], bleu_scores[-1]))
