#!/usr/bin/env python3
import sys
import string
from signal import signal, SIGPIPE, SIG_DFL

signal(SIGPIPE, SIG_DFL)

punk = '.!?:")'

def is_well_formed(line):
    if len(line) < 21:
        return False

    x = line[0]
    if not x.isdigit() and not (x.isalpha() and x.isupper()):
        return False
    if not line[-2] in punk:  # last character is '\n'
        return False

    i = 0
    k = 0

    for c in line:
        if c == ' ':
            continue

        k += 1
        if c.isalpha():
            i += 1

    j = 0
    prev = None
    for word in line.split():
        if prev is not None and word == prev:
            j += 1
            if j > 3:
                return False
        else:
            prev = word
            j = 1

    return i >= 20 and i >= k * 0.75


if __name__ == '__main__':
    for line in sys.stdin:
        if is_well_formed(line):
            sys.stdout.write(line)
