#!/usr/bin/python3

import argparse
import sys
import subprocess
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+')
parser.add_argument('--head', action='store_true')
parser.add_argument('--shuf', action='store_true')
parser.add_argument('-n', type=int)
parser.add_argument('-d', '--delimiter', default='^', choices=['&', '^', '@', '~', '|', '/', '#', '$'])
parser.add_argument('--space', action='store_true')

args = parser.parse_args()

commands = []
paste = ['paste', '-d', args.delimiter] + list(args.files)
commands.append(paste)

if args.shuf:
    shuf = ['shuf']
    if args.n:
        shuf += ['-n', str(args.n)]
    commands.append(shuf)
if args.head:
    head = ['head', '-n', str(args.n or 10)]
    commands.append(head)

if args.space:
    space = ['sed', 'G']
    commands.append(space)

delimiter = re.escape(args.delimiter) if args.delimiter in ('/', '^', '$') else args.delimiter
sed = ['sed', 's/{}/\\n/g'.format(delimiter)]
commands.append(sed)

ps = None

for i, cmd in enumerate(commands):
    stdout = sys.stdout if i == len(commands) - 1 else subprocess.PIPE
    stdin = None if i == 0 else ps.stdout
    ps = subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=open('/dev/null', 'w'))

ps.wait()

