#!/usr/bin/env python3
import argparse
import subprocess
import os
import shutil
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument('model_dir')
parser.add_argument('dest_dir')
parser.add_argument('--move', action='store_true')
parser.add_argument('--copy-data', action='store_true')
parser.add_argument('--compact', action='store_true')
parser.add_argument('--force', action='store_true')

args = parser.parse_args()


if os.path.exists(args.dest_dir):
    if args.force and os.path.isdir(args.dest_dir):
        shutil.rmtree(args.dest_dir)
    else:
        raise Exception
if not os.path.isdir(args.model_dir):
    raise Exception

config_dir = os.path.realpath(args.dest_dir)
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if config_dir.startswith(root_dir):
    config_dir = config_dir[len(root_dir):]
else:
    config_dir = args.dest_dir

if args.compact:
    os.makedirs(os.path.join(args.dest_dir, 'checkpoints'))

    files = ['config.yaml', 'default.yaml', 'log.txt', 'code.tar.gz']
    dirs = ['data']

    for filename in files:
        shutil.copy(os.path.join(args.model_dir, filename), args.dest_dir)
    for dirname in dirs:
        shutil.copytree(os.path.join(args.model_dir, dirname), os.path.join(args.dest_dir, dirname))

    checkpoint_dir = os.path.join(args.model_dir, 'checkpoints')
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('best.') or filename in ('vars.pkl', 'scores.txt'):
            shutil.copy(os.path.join(checkpoint_dir, filename),
                        os.path.join(args.dest_dir, 'checkpoints'))

    if args.move:  # delete
        shutil.rmtree(args.model_dir)
elif args.move:
    shutil.move(args.model_dir, args.dest_dir)
else:
    shutil.copytree(args.model_dir, args.dest_dir)


config_filename = os.path.join(args.dest_dir, 'config.yaml')
with open(config_filename) as f:
    content = f.read()

content = re.sub(r'model_dir:.*?\n', 'model_dir: {}\n'.format(args.dest_dir), content, flags=re.MULTILINE)
with open(config_filename, 'w') as f:
    f.write(content)

if args.copy_data:
    data_dir = re.search(r'data_dir:\s*(.*)\s*\n', content, flags=re.MULTILINE).group(1)

    content = re.sub(r'data_dir:.*?\n', 'data_dir: {}/data\n'.format(args.dest_dir), content, flags=re.MULTILINE)
    with open(config_filename, 'w') as f:
        f.write(content)

    for filename in os.listdir(data_dir):
        if filename.startswith('dev') or filename.startswith('test'):
            shutil.copy(os.path.join(data_dir, filename), os.path.join(args.dest_dir, 'data', filename))
