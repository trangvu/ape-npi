#!/usr/bin/env python3

import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('hyp')
parser.add_argument('--reference')
parser.add_argument('--source')
parser.add_argument('--max', type=int)

args = parser.parse_args()

if args.reference is not None:
    with open(args.reference) as ref_file:
        ref_lines = [line.split() for line in ref_file]
        ref_words = list(map(Counter, ref_lines))
else:
    ref_words = None
    ref_lines = None

if args.source is not None:
    with open(args.source) as src_file:
        src_lines = [line.split() for line in src_file]
else:
    src_lines = None

total = Counter()
ok = Counter()
del_counts = Counter()
ok_del_counts = Counter()

def extract_deletes(ops, src_words):
    i = 0
    deletes = []

    for op in ops:
        if op == '<KEEP>':
            i += 1
        elif op == '<DEL>':
            deletes.append(src_words[i])

    return deletes

with open(args.hyp) as hyp_file:
    for i, line in enumerate(hyp_file):
        if ref_words and i >= len(ref_words):
            break

        if src_lines and i < len(src_lines):
            hyp_del = Counter(extract_deletes(line.split(), src_lines[i]))
            del_counts += hyp_del

            if ref_lines:
                ref_del = Counter(extract_deletes(ref_lines[i], src_lines[i]))
                ok_del_counts += Counter(
                    dict((w, min(c, ref_del[w]))
                    for w, c in hyp_del.items())
                )

        words = Counter(line.split())
        total += words

        if ref_words:
            ref = ref_words[i]
            ok += Counter(dict((w, min(c, ref[w])) for w, c in words.items()))

total_count = sum(total.values())

precision_header = ' {:8}'.format('precision') if args.reference else ''
header = '{:15} {:8} {:8}'.format('word', 'count', 'percentage') + precision_header
print(header)

for w, c in total.most_common(args.max):
    precision = ' {:8.2f}%'.format(100 * ok[w] / c) if args.reference else ''

    print('{:15} {:8} {:8.2f}%'.format(w, c, 100 * c / total_count) + precision)

if del_counts:
    print('\nMost deleted words')
    for w, c in del_counts.most_common(args.max):
        precision = ' {:8.2f}%'.format(100 * ok_del_counts[w] / c) if args.source else ''

        print('{:15} {:8} {:8.2f}%'.format(w, c, 100 * c / sum(del_counts.values())) + precision)