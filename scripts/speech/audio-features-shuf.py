#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import struct
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
parser.add_argument('-n', type=int, default=0)

parser.add_argument('--input-txt', nargs='*')
parser.add_argument('--output-txt', nargs='*')

args = parser.parse_args()


with open(args.input, 'rb') as input_file:
    header = input_file.read(8)
    line_count, dim = struct.unpack('ii', header)

    indices = list(range(line_count))
    random.shuffle(indices)

    if args.n > 0:
        indices = indices[:args.n]

    frames = []

    for _ in range(line_count):
        x = input_file.read(4)
        frame_count, = struct.unpack('i', x)
        frames_ = input_file.read(4 * frame_count * dim)
        frames.append(frames_)  # this can take a lot of memory

with open(args.output, 'wb') as output_file:
    output_file.write(header)
    for index in indices:
        frames_ = frames[index]
        output_file.write(struct.pack('i', len(frames_) // (4 * dim)))
        output_file.write(frames_)

if args.input_txt and args.output_txt:
    for input_filename, output_filename in zip(args.input_txt, args.output_txt):
        with open(input_filename) as input_file, open(output_filename, 'w') as output_file:
            lines = input_file.readlines()
            for index in indices:
                line = lines[index]
                output_file.write(line)
