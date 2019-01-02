#!/usr/bin/env python3

import argparse
from translate.evaluation import tercom_statistics

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')

parser.add_argument('--case-insensitive', '-i', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.source) as src_file, open(args.target) as trg_file:
        hypotheses = [line.strip() for line in src_file]
        references = [line.strip() for line in trg_file]

        total, _ = tercom_statistics(hypotheses, references, not args.case_insensitive)

        total['TER'] = total['ERRORS'] / total['REF_WORDS']
        print(' '.join('{}={:.2f}'.format(k, v) for k, v in sorted(total.items())))
