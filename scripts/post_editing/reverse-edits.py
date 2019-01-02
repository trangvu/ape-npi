#!/usr/bin/env python3

import argparse
from translate import utils

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('edits')


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.source) as src_file, open(args.edits) as edit_file:
        for src_line, edits in zip(src_file, edit_file):
            trg_line = utils.reverse_edits(src_line.split(), [edits.split()])
            print(' '.join(trg_line))
