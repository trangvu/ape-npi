#!/usr/bin/env python3

import argparse
from collections import Counter
from itertools import starmap

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('vocab')


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.filename) as f, open(args.vocab) as vocab_file:
        vocab = set(line.strip() for line in vocab_file)

        true_vocab = Counter(w for line in f for w in line.split())

        unk_words = Counter({w: c for w, c in true_vocab.items() if w not in vocab})

        print('Unknown words:')
        print('\n'.join(starmap('  {:20} {}'.format, unk_words.most_common()[::-1])))

        print('{:22} {} ({:.2f}%)'.format('Unknown words:', len(unk_words), 100 * len(unk_words) / len(true_vocab)))

        total_unk_words = sum(unk_words.values())
        total_words = sum(true_vocab.values())
        print('{:22} {} ({:.2f}%)'.format('Total count:', total_unk_words, 100 * total_unk_words / total_words))
