#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('source_file')
parser.add_argument('target_file')
parser.add_argument('-s', '--separator', default='|||')

args = parser.parse_args()

with open(args.source_file) as src_file, open(args.target_file) as trg_file:
    for src, trg in zip(src_file, trg_file):
        line = ' '.join([src.rstrip(), args.separator, trg.rstrip()])
        print(line)

