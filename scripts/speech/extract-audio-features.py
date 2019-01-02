#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import numpy as np
import yaafelib
import struct
import sys
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs='*', help='audio filenames corresponding to one line each')
parser.add_argument('--output', dest='output_file', help='output file')
parser.add_argument('--derivatives', action='store_true')

args = parser.parse_args()

if not args.filenames:
    args.filenames = [filename.strip() for filename in sys.stdin]

parameters = dict(
    step_size=160,  # corresponds to 10 ms (at 16 kHz)
    block_size=640,  # corresponds to 40 ms
    mfcc_coeffs=40,
    mfcc_filters=41  # more filters? (needs to be at least mfcc_coeffs+1, because first coeff is ignored)
)

# TODO: ensure that all input files use this rate
fp = yaafelib.FeaturePlan(sample_rate=16000)

mfcc_features = 'MFCC MelNbFilters={mfcc_filters} CepsNbCoeffs={mfcc_coeffs} ' \
                'blockSize={block_size} stepSize={step_size}'.format(**parameters)
energy_features = 'Energy blockSize={block_size} stepSize={step_size}'.format(**parameters)

fp.addFeature('mfcc: {}'.format(mfcc_features))
if args.derivatives:
    fp.addFeature('mfcc_d1: {} > Derivate DOrder=1'.format(mfcc_features))
    fp.addFeature('mfcc_d2: {} > Derivate DOrder=2'.format(mfcc_features))

fp.addFeature('energy: {}'.format(energy_features))
if args.derivatives:
    fp.addFeature('energy_d1: {} > Derivate DOrder=1'.format(energy_features))
    fp.addFeature('energy_d2: {} > Derivate DOrder=2'.format(energy_features))

if args.derivatives:
    keys = ['mfcc', 'mfcc_d1', 'mfcc_d2', 'energy', 'energy_d1', 'energy_d2']
else:
    keys = ['mfcc', 'energy']

df = fp.getDataFlow()
engine = yaafelib.Engine()
engine.load(df)

afp = yaafelib.AudioFileProcessor()

frame_counter = Counter()

with open(args.output_file, 'wb') as f:
    for i, filename in enumerate(args.filenames):
        afp.processFile(engine, filename)
        feats_ = engine.readAllOutputs()
        feats = np.concatenate([feats_[k] for k in keys], axis=1)
        frames, dim = feats.shape
        frame_counter[frames] += 1
        feats = feats.flatten()

        if frames == 0:
            print(frames, dim, filename)
            raise Exception

        if i == 0:  # write header
            f.write(struct.pack('ii', len(args.filenames), dim))
        f.write(struct.pack('i' + 'f' * len(feats), frames, *feats))


def read_features(filename):
    all_feats = []

    with open(filename, 'rb') as f:
        lines, dim = struct.unpack('ii', f.read(8))
        for _ in xrange(lines):
            frames, = struct.unpack('i', f.read(4))
            n = frames * dim
            feats = struct.unpack('f' * n, f.read(4 * n))
            all_feats.append(np.array(feats).reshape(frames, dim))

    return all_feats
