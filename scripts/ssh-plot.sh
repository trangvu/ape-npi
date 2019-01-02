#!/usr/bin/env bash
set -e

rm -f /tmp/plot.svg
ssh $1 "rm -f /tmp/plot.svg && seq2seq/scripts/plot-loss.py ${@:2} --output /tmp/plot.svg --no-x"
scp $1:/tmp/plot.svg /tmp/ >/dev/null 2>&1
if [ -f /tmp/plot.svg ]; then
    eog /tmp/plot.svg >/dev/null 2>&1 &
fi

