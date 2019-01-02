#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('eval_dir')
parser.add_argument('reference')
parser.add_argument('--max-step', type=int)
args = parser.parse_args()

filenames = sorted(os.listdir(args.eval_dir), key=lambda filename: int(filename.split('.')[-2]))
steps = [int(filename.split('.')[-2]) for filename in filenames]

filenames = [os.path.join(args.eval_dir, filename) for filename in filenames]

with open(args.reference) as ref_file:
    lines = [line.split() for line in ref_file]
    ref_keeps = [line.count('<KEEP>') for line in lines]
    ref_dels = [line.count('<DEL>') for line in lines]
    ref_ins = [len(line) - line.count('<KEEP>') - line.count('<DEL>') for line in lines]


keeps = []
dels = []
ins = []

fun = lambda x, y, z: abs(x - y) / z
#fun = lambda x, y, z: x/z

for filename in filenames:
    with open(filename) as f:
        keep_ = 0
        del_ = 0
        ins_ = 0
        lines = 0

        for i, line in enumerate(f):
            words = line.split()
            lines += 1
            keep_ += fun(words.count('<KEEP>'), ref_keeps[i], len(words))
            del_ += fun(words.count('<DEL>'), ref_dels[i], len(words))
            ins_ += fun(len(words) - words.count('<KEEP>') - words.count('<DEL>'), ref_ins[i], len(words))

        keeps.append(keep_ / lines)
        dels.append(del_ / lines)
        ins.append(ins_ / lines)


if args.max_step:
    steps, keeps, dels, ins = zip(*[
        (step, keep_, del_, ins_) for step, keep_, del_, ins_
        in zip(steps, keeps, dels, ins) if step <= args.max_step
    ])

plt.plot(steps, keeps, label='KEEP')
plt.plot(steps, dels, label='DEL')
plt.plot(steps, ins, label='INS(x)')

legend = plt.legend(loc='best', shadow=True)

plt.show()