#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('vocab')
parser.add_argument('bpe')


def build_vocab(bpe_pairs):
    vocab = set()
    for a, b in bpe_pairs:
        words = [a, b, a + b]
        for word in words:
            if word.endswith('</w>'):
                vocab.add(word[:-4])
            else:
                vocab.add(word + '@@')
                vocab.add(word)
    return vocab


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.bpe) as bpe_file, open(args.vocab) as vocab_file:
        bpe_pairs = [line.split() for line in bpe_file]
        vocab = [line.strip() for line in vocab_file]

        bpe_vocab = build_vocab(bpe_pairs)

        for w in vocab:
            print(w)

        vocab = set(vocab)
        for w in bpe_vocab:
            if w not in vocab:
                print(w)
