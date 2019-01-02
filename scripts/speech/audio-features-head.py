#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import struct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-n', type=int, default=10)

args = parser.parse_args()

with open(args.input, 'rb') as input_file, open(args.output, 'wb') as output_file:
    header = input_file.read(8)
    lines, dim = struct.unpack('ii', header)
    lines = min(lines, args.n)
    output_file.write(struct.pack('ii', lines, dim))

    for _ in range(lines):
        x = input_file.read(4)
        frames, = struct.unpack('i', x)

        output_file.write(x)
        output_file.write(input_file.read(4 * frames * dim))

