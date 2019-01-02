#!/usr/bin/env python3

import argparse
import numpy as np
from translate.evaluation import tercom_statistics
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('target')
parser.add_argument('--output')
parser.add_argument('--precision', type=int, default=4)

parser.add_argument('--case-insensitive', '-i', action='store_true')

if __name__ == '__main__':
    args = parser.parse_args()

    vectors = []
    fields = ['DEL', 'INS', 'SUB', 'WORD_SHIFT', 'REF_WORDS', 'TER']

    with open(args.source) as src_file, open(args.target) as trg_file:

        i = 0
        n = 1000

        avg_length = 0

        while True:
            i += 1
            hypotheses = list(islice(src_file, n))
            references = list(islice(trg_file, n))

            if not hypotheses or not references:
                break

            hypotheses = [line.strip() for line in hypotheses]
            references = [line.strip() for line in references]

            _, stats = tercom_statistics(hypotheses, references, not args.case_insensitive)

            if avg_length == 0:
                avg_length = sum(stats_['REF_WORDS'] for stats_ in stats) / len(stats)

            for stats_ in stats:
                for field in ('DEL', 'INS', 'SUB', 'WORD_SHIFT'):
                    stats_[field] /= stats_['REF_WORDS']

                stats_['REF_WORDS'] = (stats_['REF_WORDS'] - avg_length) / avg_length
                stats_['TER'] /= 100

            if not args.output:
                print('\n'.join(','.join(str(round(stats_[k], args.precision)) for k in fields)
                                for stats_ in stats))
            else:
                vectors += [np.array([stats_[k] for k in fields]) for stats_ in stats]
                print('{}'.format(i * n), end='\r')

        if args.output:
            import h5py
            h5f = h5py.File(args.output, 'w')
            h5f.create_dataset('dataset_1', data=vectors)
            h5f.close()
