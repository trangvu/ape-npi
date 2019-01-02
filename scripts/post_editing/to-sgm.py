#!/usr/bin/env python3

import argparse
import sys
import numpy as np
#from translate.evaluation import corpus_bleu, corpus_ter

parser = argparse.ArgumentParser()
# parser.add_argument('source1')
# parser.add_argument('source2')
# parser.add_argument('target')
#
# parser.add_argument('--bleu', action='store_true')
# parser.add_argument('--max-size', type=int)
# parser.add_argument('--case-insensitive', '-i', action='store_true')
#
# parser.add_argument('--draws', type=int, default=1000)
# parser.add_argument('--sample-size', type=int, default=0)
# parser.add_argument('-p', type=float, default=0.05)
parser.add_argument('--set-type')
parser.add_argument('--set-id')

args = parser.parse_args()

if args.set_type is not None:
    if args.set_id is None:
        args.set_id = 'dummy'

    print('<{} setid="{}" srclang="any" trglang="any">'.format(args.set_type, args.set_id))

print('<doc docid="dummy" sysid="{}">'.format(args.set_type))
for i, line in enumerate(sys.stdin, 1):
    print('<seg id="{}">{}</seg>'.format(i, line.strip()))
print('</doc>')

if args.set_type is not None:
    print('</{}>'.format(args.set_type))
