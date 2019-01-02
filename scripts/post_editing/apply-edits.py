#!/usr/bin/env python3

import argparse

_BOS = '<S>'
_EOS = '</S>'
_UNK = '<UNK>'
_KEEP = '<KEEP>'
_DEL = '<DEL>'
_INS = '<INS>'
_SUB = '<SUB>'
_NONE = '<NONE>'

def reverse_edits(source, edits, fix=True, strict=False):
    if len(edits) == 1:    # transform list of edits as a list of (op, word) tuples
        edits = edits[0]
        for i, edit in enumerate(edits):
            if edit in (_KEEP, _DEL, _INS, _SUB):
                edit = (edit, edit)
            elif edit.startswith(_INS + '_'):
                edit = (_INS, edit[len(_INS + '_'):])
            elif edit.startswith(_SUB + '_'):
                edit = (_SUB, edit[len(_SUB + '_'):])
            else:
                edit = (_INS, edit)

            edits[i] = edit
    else:
        edits = zip(*edits)

    src_words = source
    target = []
    consistent = True
    i = 0

    for op, word in edits:
        if strict and not consistent:
            break
        if op in (_DEL, _KEEP, _SUB):
            if i >= len(src_words):
                consistent = False
                continue

            if op == _KEEP:
                target.append(src_words[i])
            elif op == _SUB:
                target.append(word)

            i += 1
        else:   # op is INS
            target.append(word)

    if fix:
        target += src_words[i:]

    return target



parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('edits')
parser.add_argument('--not-strict', action='store_false', dest='strict')
parser.add_argument('--no-fix', action='store_false', dest='fix')

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.source) as src_file, open(args.edits) as edit_file:
        for source, edits in zip(src_file, edit_file):
            target = reverse_edits(source.strip('\n').split(), [edits.strip('\n').split()], strict=args.strict, fix=args.fix)
            print(' '.join(target))
