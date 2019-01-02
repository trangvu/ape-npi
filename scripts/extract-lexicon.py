#!/usr/bin/python3

import argparse
from collections import defaultdict, OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('source_file')
parser.add_argument('target_file')
parser.add_argument('align_file')

args = parser.parse_args()

src_vocab = OrderedDict()
trg_vocab = OrderedDict()

counts = defaultdict(dict)

with open(args.source_file) as src_file, open(args.target_file) as trg_file, open(args.align_file) as align_file:
    for src, trg, align in zip(src_file, trg_file, align_file):
        src = src.split()
        trg = trg.split()
        align = align.split()
        for i, j in map(lambda p: map(int, p.split('-')), align):
            src_ = src[i]
            trg_ = trg[j]

            src_id = src_vocab.setdefault(src_, len(src_vocab))
            trg_id = trg_vocab.setdefault(trg_, len(trg_vocab))

            #src_counts[src_id] = src_counts.get(src_id, 0) + 1
            #trg_counts[trg_id] = trg_counts.get(trg_id, 0) + 1
            #pair_counts((src_id, trg_id)) = pair_counts.get((src_id, trg_id), 0) + 1
            
            counts[src_id][trg_id] = counts[src_id].get(trg_id, 0) + 1

src_vocab = list(src_vocab.keys())
trg_vocab = list(trg_vocab.keys())

for source, counts_ in counts.items():
    target = max(counts_.keys(), key=lambda word: counts_[word])
    source = src_vocab[source]
    target = trg_vocab[target]
    print(source, target)
