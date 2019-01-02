#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import struct
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+')
parser.add_argument('output')

args = parser.parse_args()

with open(args.output, 'wb') as output_file:
    lines = 0
    dim = None
    for filename in args.inputs:
        with open(filename, 'rb') as input_file:
            header = input_file.read(8)
            lines_, dim_ = struct.unpack('ii', header)
            lines += lines_
            if dim is not None and dim_ != dim:
                raise Exception('incompatible dimensions')
            dim = dim_

    output_file.write(struct.pack('ii', lines, dim))

    for filename in args.inputs:
        with open(filename, 'rb') as input_file:
            header = input_file.read(8)
            lines_, dim_ = struct.unpack('ii', header)

            for _ in range(lines_):
                x = input_file.read(4)
                frames, = struct.unpack('i', x)

                output_file.write(x)
                output_file.write(input_file.read(4 * frames * dim))
