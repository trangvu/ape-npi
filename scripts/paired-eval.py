#!/usr/bin/env python3

import argparse
import sys
import numpy as np
from translate.evaluation import corpus_bleu, corpus_ter

parser = argparse.ArgumentParser()
parser.add_argument('source1')
parser.add_argument('source2')
parser.add_argument('target')

parser.add_argument('--bleu', action='store_true')
parser.add_argument('--max-size', type=int)
parser.add_argument('--case-insensitive', '-i', action='store_true')

parser.add_argument('--samples', type=int, default=1000)
parser.add_argument('--sample-size', type=int, default=0)
parser.add_argument('-p', type=float, default=0.05)


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.source1) as src_file_1, open(args.source2) as src_file_2, open(args.target) as trg_file:
        if args.case_insensitive:
            fun = lambda x: x.strip().lower()
        else:
            fun = lambda x: x.strip()

        hypotheses_1 = list(map(fun, src_file_1))
        hypotheses_2 = list(map(fun, src_file_2))
        references = list(map(fun, trg_file))

        if args.max_size is not None:
            hypotheses_1 = hypotheses_1[:args.max_size]
            hypotheses_2 = hypotheses_2[:args.max_size]
            references = references[:args.max_size]

        if len(hypotheses_1) != len(references) or len(hypotheses_2) != len(references):
            sys.stderr.write('warning: source and target don\'t have the same length\n')
            size = min(len(hypotheses_1), len(hypotheses_2), len(references))
            hypotheses_1 = hypotheses_1[:size]
            hypotheses_2 = hypotheses_2[:size]
            references = references[:size]

        indices = np.arange(len(references))
        if args.sample_size == 0:
            args.sample_size = len(references)

        diffs = []

        hypotheses_1 = np.array(hypotheses_1)
        hypotheses_2 = np.array(hypotheses_2)
        references = np.array(references)

        score_fun = corpus_bleu if args.bleu else corpus_ter

        #diff = abs(score_fun(hypotheses_1, references)[0] - score_fun(hypotheses_2, references)[0])

        for _ in range(args.samples):
            indices = np.random.randint(len(references), size=args.sample_size)
            hypotheses_1_ = hypotheses_1[indices]
            hypotheses_2_ = hypotheses_2[indices]
            references_ = references[indices]

            score_1, _ = score_fun(hypotheses_1_, references_)
            score_2, _ = score_fun(hypotheses_2_, references_)

            diffs.append(int(score_1 > score_2))
            #diffs.append(abs(score_1 - score_2))

        # avg_diff = sum(diffs) / len(diffs)
        # c = sum(
        #     int(diff_ - avg_diff >= diff) for diff_ in diffs
        # )
        #
        # p = (c + 1) / (len(diffs) + 1)
        # print(p)

        p = sum(diffs) / len(diffs)
        if not args.bleu:
            p = 1 - p

        print('x is better than y {:.1f}% of the time'.format(p * 100))
