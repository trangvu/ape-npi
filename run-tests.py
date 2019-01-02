#!/usr/bin/env python3
import subprocess
import shlex
import re
import os
import argparse

OKGREEN = '\033[92m'
FAIL = '\033[91m'
ENDC = '\033[0m'

parser = argparse.ArgumentParser()
parser.add_argument('--no-gpu', action='store_true')
parser.add_argument('--gpu-id', type=int)
parser.add_argument('dirs', nargs='*')
args = parser.parse_args()

extra_params = []
if args.no_gpu:
    extra_params.append('--no-gpu')
if args.gpu_id is not None:
    extra_params += ['--gpu-id', str(args.gpu_id)]

def failure(message):
    print('{}failure: {}{}'.format(FAIL, message, ENDC))
def success(message):
    print('{}success: {}{}'.format(OKGREEN, message, ENDC))

log_file = os.path.join('tests', 'log.txt')

try:
    os.remove(log_file)
except FileNotFoundError:
    pass


def get_best_score(log_file):
    scores = []
    with open(log_file) as f:
        for line in f:
            score_ = re.search(r' (score|bleu|ter|loss|cer|wer|bleu1)=(.*?) ', line + ' ')

            if score_:
                scores.append(float(score_.group(2)))

    if len(scores) == 0:
        return None
    elif len(scores) == 1:
        return scores[0]
    elif scores[0] <= scores[-1]:
        return max(scores)
    else:
        return min(scores)


def run(dir_, score=None):
    config_file = os.path.join(dir_, 'config.yaml')
    log_file_ = os.path.join(dir_, 'log.txt')
    name = os.path.basename(dir_)

    if score is None:
        try:
            score = get_best_score(log_file_)
        except:
            pass

    print('Running {}'.format(name))

    try:
        output = subprocess.check_output(['./seq2seq.sh', config_file, '--eval'] + extra_params,
                                         stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        output = e.output.decode()

    scores = output.strip().split('\n')[-1] + ' '
    score_ = re.search(r' (score|bleu|ter|loss|cer|wer|bleu1)=(.*?) ', scores)

    with open(log_file, 'a') as f:
        f.write(output)

    if not score_:
        failure('unable to run test (see log file)')
    else:
        score_ = float(score_.group(2)) 
        if score is None:
            success('obtained {}'.format(score_))
        elif score_ == score:
            success('scores matching ({})'.format(score_))
        else:
            failure('obtained {}, expected {}'.format(score_, score))


if not args.dirs:
    dirs = [os.path.join('tests', name) for name in os.listdir('tests')]
else:
    dirs = args.dirs

for path in dirs:
    if os.path.isdir(path):
        run(path)

