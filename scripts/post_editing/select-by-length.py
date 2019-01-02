#!/usr/bin/env python3

import argparse
import random
import sys

parser = argparse.ArgumentParser()

parser.add_argument('ref_sentences')
parser.add_argument('sentences')
parser.add_argument('-n', type=int, default=500000)
parser.add_argument('-k', type=int, default=1)
parser.add_argument('-m', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.ref_sentences) as f:
        ref_lengths = [len(line.split()) for line in f]
    with open(args.sentences) as f:
        lengths = [len(line.split()) for line in f]
        lengths = list(enumerate(lengths))

    n = 0
    l = len(lengths)

    while n < args.n and l > 0:
        length = ref_lengths[n % len(ref_lengths)]

        def key(i):
            return abs(length - lengths[i][1])

        indices = random.sample(range(l), k=args.m)

        if args.k > 1:
            indices = sorted(indices, key=key)[:args.k]
        else:
            indices = [min(indices, key=key)]

        for i in indices:
            sys.stdout.write(str(lengths[i][0]) + '\n')

        #sys.stdout.flush()

        for i in indices:
            lengths[i], lengths[l - 1] = lengths[l - 1], lengths[i]
            l -= 1
            n += 1
