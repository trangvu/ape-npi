#!/usr/bin/env python3

import argparse
import sys
import sklearn.mixture
import numpy as np
import random
from scipy.stats import truncnorm
from collections import Counter
from translate.evaluation import tercom_statistics

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')

parser.add_argument('--mono')
parser.add_argument('--min-count', type=int, default=2)
parser.add_argument('--case-insensitive', '-i', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    fields = ['DEL', 'INS', 'SUB', 'WORD_SHIFT', 'REF_WORDS']
    op_fields = ['DEL', 'INS', 'SUB', 'WORD_SHIFT']

    with open(args.source) as src_file, open(args.target) as trg_file:
        hypotheses = [line.strip() for line in src_file]
        references = [line.strip() for line in trg_file]

        _, stats = tercom_statistics(hypotheses, references, not args.case_insensitive)

        for stats_ in stats:
            for field in op_fields:
                stats_[field] /= stats_['REF_WORDS']

        ops = np.array([[stats_[k] for k in op_fields] for stats_ in stats])

        model = sklearn.mixture.GMM(n_components=1)
        model.fit(ops)

        sigma = model.covars_
        mu = model.means_
        distribution = truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma)

    unigram_filename = args.mono or args.source
    with open(unigram_filename) as unigram_file:
        unigrams = Counter(w for line in unigram_file for w in line.split())
        unigrams = Counter({w: c for w, c in unigrams.items() if c >= args.min_count})

        total = sum(unigrams.values())
        for k in unigrams.keys():
            unigrams[k] /= total

    vocab = list(unigrams.keys())
    p = np.array(list(unigrams.values()))

    def unigram_sampler():
        while True:
            x = np.random.choice(vocab, size=1000, p=p)
            for w in x:
                yield w

    sampler = unigram_sampler()

    for line in sys.stdin:
        words = line.split()

        sample = distribution.rvs(len(op_fields)) * len(words)

        x = sample.astype(np.int32)
        i = np.random.random(sample.shape) < sample - sample.astype(np.int32)
        x += i.astype(np.int32)

        dels, ins, subs, shifts = x

        for _ in range(dels):
            k = random.randrange(len(words))
            del words[k]

        for _ in range(shifts):
            j, k = random.sample(range(len(words)), 2)
            w = words.pop(j)
            words.insert(k, w)

        for _ in range(subs):
            w = next(sampler)
            k = random.randrange(len(words))
            words[k] = w

        for _ in range(ins):
            w = next(sampler)
            k = random.randrange(len(words) + 1)
            words.insert(k, w)

        print(' '.join(words))
