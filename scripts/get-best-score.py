#!/usr/bin/env python3

import itertools
import argparse
import re
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('log_files', nargs='+')
parser.add_argument('--dev-prefix')
parser.add_argument('--score', choices=('ter', 'bleu', 'wer'))
parser.add_argument('--task-name')
parser.add_argument('--time', action='store_true')
parser.add_argument('--params', action='store_true')

parser.add_argument('--ter', action='store_true')
parser.add_argument('--bleu', action='store_true')
parser.add_argument('--wer', action='store_true')
parser.add_argument('--cer', action='store_true')
parser.add_argument('--bleu1', action='store_true')
parser.add_argument('--loss', action='store_true')


def print_scores(log_file, time=False, label=None):
    with open(log_file) as log_file:
        scores = {}
        times = {}
        current_step = 0
        max_step = 0
        starting_time = None
        param_count = None

        def read_time(line):
            if not time:
                return None
            m = re.match('../.. ..:..:..', line)
            if m:
                return dateutil.parser.parse(m.group(0))

        for line in log_file:
            if starting_time is None:
                starting_time = read_time(line)
            if param_count is None:
                m = re.search('number of parameters: (.*)', line)
                if m:
                    param_count = m.group(1)

            m = re.search('step (\d+)', line)
            if m:
                current_step = int(m.group(1))
                times.setdefault(current_step, read_time(line)) 
                max_step = max(max_step, current_step)
                continue

            if args.task_name is not None:
                if not re.search(args.task_name, line):
                    continue
            if args.dev_prefix is not None:
                if not re.search(args.task_name, line):
                    continue

            m = re.findall('(loss|bleu|score|ter|wer|cer|bleu1|penalty|ratio)=(\d+.\d+)', line)
            if m:
                scores_ = {k: float(v) for k, v in m}
                scores.setdefault(current_step, scores_)

        def key(d):
            score = d.get(args.score.lower())
            if score is None:
                score = d.get('score')

            if args.score in ('ter', 'wer', 'cer', 'loss'):
                score = -score
            return score

        step, best = max(scores.items(), key=lambda p: key(p[1]))

        if 'score' in best:
            missing_key = next(k for k in ['bleu', 'ter', 'wer', 'cer', 'bleu1', 'loss'] if k not in best)
            best[missing_key] = best.pop('score')

        keys = [args.score, 'bleu', 'ter', 'wer', 'cer', 'bleu1', 'loss', 'penalty', 'ratio']
        best = sorted(best.items(), key=lambda p: keys.index(p[0]))

        def pretty_time(seconds):
            seconds = int(seconds)
            s = []
            days, seconds = divmod(seconds, 3600 * 24)
            if days > 0:
                s.append('{}d'.format(days))
            hours, seconds = divmod(seconds, 3600)
            if hours > 0:
                s.append('{}h'.format(hours))
            minutes, seconds = divmod(seconds, 60)
            if minutes > 0 and days == 0:
                s.append('{}min'.format(minutes))
            if days == 0 and hours == 0 and minutes < 5 and seconds > 0:
                s.append('{}s'.format(seconds))
            return ''.join(s)

        if time:
            total_time = (times[max_step] - starting_time).total_seconds()
            train_time = (times[step] - starting_time).total_seconds()
            time_string = ' time={}/{}'.format(pretty_time(train_time), pretty_time(total_time))
        else:
            time_string = ''

        if label is None:
            label = ''
        if args.params and param_count is not None:
            param_string = ' params={}'.format(param_count)
        else:
            param_string = ''

        print(label + ' '.join(itertools.starmap('{}={:.2f}'.format, best)),
              'step={}/{}'.format(step, max_step) + param_string + time_string)

if __name__ == '__main__':
    args = parser.parse_args()
    args.log_files = [os.path.join(log_file, 'log.txt') if os.path.isdir(log_file) else log_file
                      for log_file in args.log_files]

    if args.ter:
        args.score = 'ter'
    elif args.wer:
        args.score = 'wer'
    elif args.cer:
        args.score = 'cer'
    elif args.bleu1:
        args.score = 'bleu1'
    elif args.loss:
        args.score = 'loss'
    elif args.bleu or args.score is None:
        args.score = 'bleu'

    if args.time:
        import dateutil.parser

    labels = None
    if not labels:
        filenames = [os.path.basename(log_file) for log_file in args.log_files]
        if len(set(filenames)) == len(filenames):
            labels = filenames
    if not labels:
        dirnames = [os.path.basename(os.path.dirname(log_file)) for log_file in args.log_files]
        if len(set(dirnames)) == len(dirnames):
            labels = dirnames
    if not labels:
        labels = ['model_{}'.format(i) for i in range(1, len(args.log_files) + 1)]

    label_len = max(map(len, labels))
    format_string = '{{:<{}}}'.format(label_len + 2)

    for label, log_file in zip(labels, args.log_files):
        try:
            if len(args.log_files) == 1:
                label = None
            else:
                label = format_string.format(label + ':')

            print_scores(log_file, time=args.time, label=label)
        except:
            pass

