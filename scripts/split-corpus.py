#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('dest')
parser.add_argument('--splits', type=int, required=True)
parser.add_argument('--tokens', action='store_true')

args = parser.parse_args()

os.makedirs(args.dest, exist_ok=True)

with open(args.filename) as input_file:
    if args.tokens:
        total_size = sum(len(line.split()) for line in input_file)
    else:
        total_size = sum(1 for line in input_file)

    input_file.seek(0)

    shard_size = total_size // args.splits

    for i in range(args.splits):
        filename = os.path.join(args.dest, str(i + 1).zfill(len(str(args.splits))))

        with open(filename, 'w') as output_file:

            this_size = 0
            for line in input_file:
                line_size = len(line.split()) if args.tokens else 1
                this_size += line_size

                output_file.write(line)

                if this_size >= shard_size and i < args.splits - 1:
                    break
