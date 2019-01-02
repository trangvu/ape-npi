#!/usr/bin/env python3
import argparse
from collections import Counter, namedtuple, OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('--lower', action='store_true')
parser.add_argument('--count-whitespaces', action='store_true')
parser.add_argument('-c', '-b', '--chars', action='store_true', help='display char info')
parser.add_argument('-l', '--lines', action='store_true', help='display line count')
parser.add_argument('-w', '--words', action='store_true', help='display word info')
parser.add_argument('-a', '--all', action='store_true',
                    help='display all info and more (large memory usage)')

args = parser.parse_args()

if not args.chars and not args.lines and not args.words or args.all:
    args.chars = args.words = args.lines = True

word_counts = Counter()
char_counts = Counter()

word_dict = Counter()
char_dict = Counter()

line_dict = Counter()
lines = 0

with open(args.filename) as f:
    for line in f:
        if args.lower:
            line = line.lower()

        if args.words:
            words = line.split()
            word_counts[len(words)] += 1
            for word in words:
                word_dict[word] += 1

        if args.chars:
            chars = line
            if not args.count_whitespaces:
                chars = line.strip().replace(' ', '')

            char_counts[len(chars)] += 1
            for char in chars:
                char_dict[char] += 1

        lines += 1
        if args.all:
            line_dict[line] += 1


def info_dict(title, counter):
    total = sum(counter.values())
    unique = len(counter)
    avg = total / unique
    min_ = min(counter.values())
    max_ = max(counter.values())

    cumulative_count = 0
    coverage = OrderedDict([(90, 0), (95, 0), (99, 0)])

    for i, pair in enumerate(counter.most_common(), 1):
        _, count = pair
        cumulative_count += count

        for percent, count in coverage.items():
            if count == 0 and cumulative_count * 100 >= percent * total:
                coverage[percent] = i

    summary = [
        '{}\n{}'.format(title, '-' * len(title)),
        'Total:   {}'.format(total),
        'Unique:  {}'.format(unique),
        'Minimum: {}'.format(min_),
        'Maximum: {}'.format(max_),
        'Average: {:.1f}'.format(avg)
    ]

    for percent, count in coverage.items():
        summary.append('{}% cov: {}'.format(percent, count))

    return '\n  '.join(summary) + '\n'


def info_lengths(title, counter):
    total = sum(counter.values())
    avg = sum(k * v for k, v in counter.items()) / total

    coverage = OrderedDict([(1, 0), (5, 0), (10, 0),
                            (50, 0), (90, 0), (95, 0), (99, 0)])

    cumulative_count = 0
    prev_k = 0

    for k, v in sorted(counter.items()):
        cumulative_count += v

        for percent, count in coverage.items():
            if count == 0 and cumulative_count * 100 >= percent * total:
                coverage[percent] = prev_k if percent < 50 else k

        prev_k = k

    summary = [
        '{}\n{}'.format(title, '-' * len(title)),
        'Minimum: {}'.format(min(counter)),
        'Maximum: {}'.format(max(counter)),
        'Average: {:.1f}'.format(avg),
    ]

    for percent, count in coverage.items():
        summary.append('{}{:2d}%:   {}'.format('<=' if percent < 50 else '>=', percent, count))

    return '\n  '.join(summary) + '\n'


if args.lines:
    print("Lines\n-----\n  Total:   {}".format(lines))

if args.all:
    summary = [
        'Unique:  {}'.format(len(line_dict)),
        'Average: {:.2f}'.format(lines / len(line_dict))
    ]
    print('  ' + '\n  '.join(summary))

print()

if args.words:
    print(info_lengths('Words per line', word_counts))
    print(info_dict('Words', word_dict))

if args.chars:
    print(info_lengths('Chars per line', char_counts))
    print(info_dict('Chars', char_dict))
