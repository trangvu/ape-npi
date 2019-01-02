import functools
import subprocess
import tempfile
import math
import numpy as np
import re
import os
import random

from collections import Counter, OrderedDict


def levenshtein(src, trg, sub_cost=1.0, del_cost=1.0, ins_cost=1.0, randomize=True):
    DEL, INS, KEEP, SUB  = range(4)
    op_names = 'delete', 'insert', 'keep', 'sub'

    costs = np.zeros((len(trg) + 1, len(src) + 1))
    ops = np.zeros((len(trg) + 1, len(src) + 1), dtype=np.int32)

    costs[0] = range(len(src) + 1)
    costs[:,0] = range(len(trg) + 1)
    ops[0] = DEL
    ops[:,0] = INS

    if randomize:
        key = lambda p: (p[0], random.random())
    else:
        key = None

    for i in range(1, len(trg) + 1):
        for j in range(1, len(src) + 1):
            c, op = (sub_cost, SUB) if trg[i - 1] != src[j - 1] else (0, KEEP)
            costs[i,j], ops[i,j] = min([
                (costs[i, j - 1] + del_cost, DEL),
                (costs[i - 1, j] + ins_cost, INS),
                (costs[i - 1, j - 1] + c, op),
            ], key=key)

    # backtracking
    i, j = len(trg), len(src)
    cost = costs[i, j]

    res = []

    while i > 0 or j > 0:
        op = ops[i, j]
        op_name = op_names[op]

        if op == DEL:
            res.append(op_name)
            j -= 1
        else:
            res.append((op_name, trg[i - 1]))
            i -= 1
            if op != INS:
                j -= 1

    return cost, res[::-1]


def sentence_bleu(hypothesis, reference, smoothing=True, order=4, **kwargs):
    """
    Compute sentence-level BLEU score between a translation hypothesis and a reference.

    :param hypothesis: list of tokens or token ids
    :param reference: list of tokens or token ids
    :param smoothing: apply smoothing (recommended, especially for short sequences)
    :param order: count n-grams up to this value of n.
    :param kwargs: additional (unused) parameters
    :return: BLEU score (float)
    """
    log_score = 0

    if len(hypothesis) == 0:
        return 0

    for i in range(order):
        hyp_ngrams = Counter(zip(*[hypothesis[j:] for j in range(i + 1)]))
        ref_ngrams = Counter(zip(*[reference[j:] for j in range(i + 1)]))

        numerator = sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())
        denominator = sum(hyp_ngrams.values())

        if smoothing:
            numerator += 1
            denominator += 1

        score = numerator / denominator

        if score == 0:
            log_score += float('-inf')
        else:
            log_score += math.log(score) / order

    bp = min(1, math.exp(1 - len(reference) / len(hypothesis)))

    return math.exp(log_score) * bp


def score_function_decorator(reversed=False):
    def decorator(func):
        func.reversed = reversed
        return func
    return decorator


def divide(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.true_divide(x, y)
        z[~ np.isfinite(z)] = 0
    return z


def corpus_bleu(hypotheses, references, smoothing=False, order=4, **kwargs):
    """
    Computes the BLEU score at the corpus-level between a list of translation hypotheses and references.
    With the default settings, this computes the exact same score as `multi-bleu.perl`.

    All corpus-based evaluation functions should follow this interface.

    :param hypotheses: list of strings
    :param references: list of strings
    :param smoothing: apply +1 smoothing
    :param order: count n-grams up to this value of n. `multi-bleu.perl` uses a value of 4.
    :param kwargs: additional (unused) parameters
    :return: score (float), and summary containing additional information (str)
    """
    total = np.zeros((order,))
    correct = np.zeros((order,))

    hyp_length = 0
    ref_length = 0

    for hyp, ref in zip(hypotheses, references):
        if isinstance(ref, str):
            ref = [ref]

        hyp = hyp.split()
        ref = [ref_.split() for ref_ in ref]

        hyp_length += len(hyp)
        ref_length += min(map(len, ref), key=lambda l: (abs(l - len(hyp)), l))

        for i in range(order):
            ref_ngrams = Counter()
            for ref_ in ref:
                c = Counter(zip(*[ref_[j:] for j in range(i + 1)]))
                for ngram, count in c.items():
                    ref_ngrams[ngram] = max(count, ref_ngrams[ngram])

            hyp_ngrams = Counter(zip(*[hyp[j:] for j in range(i + 1)]))

            total[i] += sum(hyp_ngrams.values())
            correct[i] += sum(min(count, ref_ngrams[bigram]) for bigram, count in hyp_ngrams.items())

    if smoothing:
        total += 1
        correct += 1

    scores = divide(correct, total)

    score = math.exp(
        sum(math.log(score) if score > 0 else float('-inf') for score in scores) / order
    )

    bp = min(1, math.exp(1 - ref_length / hyp_length)) if hyp_length > 0 else 0.0
    bleu = 100 * bp * score

    return bleu, 'penalty={:.3f} ratio={:.3f}'.format(bp, hyp_length / ref_length)


@score_function_decorator(reversed=True)
def corpus_ter(hypotheses, references, case_sensitive=True, tercom_path=None, **kwargs):
    tercom_path = tercom_path or 'scripts/tercom.jar'

    with tempfile.NamedTemporaryFile('w') as hypothesis_file, tempfile.NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        cmd = ['java', '-jar', tercom_path, '-h', hypothesis_file.name, '-r', reference_file.name]
        if case_sensitive:
            cmd.append('-s')

        output = subprocess.check_output(cmd).decode()

        error = re.findall(r'Total TER: (.*?) ', output, re.MULTILINE)[0]
        return float(error) * 100, ''


@score_function_decorator(reversed=True)
def corpus_wer(hypotheses, references, char_based=False, **kwargs):
    def split(s):
        return tuple(s) if char_based else tuple(s.split())

    scores = [
        levenshtein(split(hyp), split(ref))[0] / len(split(ref))
        for hyp, ref in zip(hypotheses, references)
    ]

    score = 100 * sum(scores) / len(scores)

    hyp_length = sum(len(hyp.split()) for hyp in hypotheses)
    ref_length = sum(len(ref.split()) for ref in references)

    return score, 'ratio={:.3f}'.format(hyp_length / ref_length)


@score_function_decorator(reversed=True)
def corpus_cer(hypotheses, references, **kwargs):
    return corpus_wer(hypotheses, references, char_based=True)


@score_function_decorator(reversed=False)
def corpus_bleu1(hypotheses, references, **kwargs):
    return corpus_bleu(hypotheses, references, order=1)


def corpus_scores(hypotheses, references, main='bleu', **kwargs):
    bleu_score, summary = corpus_bleu(hypotheses, references)
    # ter, _ = corpus_ter(hypotheses, references)
    try:
        ter, _ = corpus_ter(hypotheses, references)
    except:  # Java not installed
        ter = 0.0

    wer, _ = corpus_wer(hypotheses, references)
    cer, _ = corpus_cer(hypotheses, references)
    bleu1, _ = corpus_bleu1(hypotheses, references)

    scores = OrderedDict([('bleu', bleu_score), ('ter', ter), ('wer', wer), ('bleu1', bleu1), ('cer', cer)])

    if main is not None:
        main_score = scores[main]
    else:
        main_score = None

    summary = ' '.join(['{}={:.2f}'.format(k, v) for k, v in scores.items() if k != main] + [summary])
    return main_score, summary


@score_function_decorator(reversed=True)
def corpus_scores_ter(*args, **kwargs):
    return corpus_scores(*args, main='ter', **kwargs)


@score_function_decorator(reversed=True)
def corpus_scores_wer(*args, **kwargs):
    return corpus_scores(*args, main='wer', **kwargs)


corpus_scores_bleu = corpus_scores


@functools.lru_cache(maxsize=1024)
def levenshtein_rec(src, trg):
    # Dynamic programming by memoization
    if len(src) == 0:
        return len(trg)
    elif len(trg) == 0:
        return len(src)

    return min(
        int(src[0] != trg[0]) + levenshtein_rec(src[1:], trg[1:]),
        1 + levenshtein_rec(src[1:], trg),
        1 + levenshtein_rec(src, trg[1:])
    )


def tercom_statistics(hypotheses, references, case_sensitive=True, **kwargs):
    with tempfile.NamedTemporaryFile('w') as hypothesis_file, tempfile.NamedTemporaryFile('w') as reference_file:
        for i, (hypothesis, reference) in enumerate(zip(hypotheses, references)):
            hypothesis_file.write('{} ({})\n'.format(hypothesis, i))
            reference_file.write('{} ({})\n'.format(reference, i))
        hypothesis_file.flush()
        reference_file.flush()

        filename = tempfile.mktemp()

        cmd = ['java', '-jar', 'scripts/tercom.jar', '-h', hypothesis_file.name, '-r', reference_file.name,
               '-o', 'sum', '-n', filename]
        if case_sensitive:
            cmd.append('-s')

        output = open('/dev/null', 'w')
        subprocess.call(cmd, stdout=output, stderr=output)

    with open(filename + '.sum') as f:
        fields = ['DEL', 'INS', 'SUB', 'SHIFT', 'WORD_SHIFT', 'ERRORS', 'REF_WORDS', 'TER']

        stats = []
        for line in f:
            values = line.strip().split('|')
            if len(values) != 9:
                continue
            try:
                # values = np.array([float(x) for x in values[1:]])
                values = dict(zip(fields, [float(x.replace(',', '.')) for x in values[1:]]))
            except ValueError:
                continue

            stats.append(values)

        assert len(stats) == len(hypotheses) + 1

        total = stats[-1]
        stats = stats[:-1]
        total = {k: v / len(stats) for k, v in total.items()}

    os.remove(filename + '.sum')

    return total, stats

name_mapping = {
    'corpus_bleu': ['bleu', 'loss'],
    'corpus_ter': ['ter', 'loss'],
    'corpus_wer': ['wer', 'loss'],
    'corpus_bleu1': ['bleu1', 'loss'],
    'corpus_cer': ['cer', 'loss'],
    'corpus_scores': ['bleu', 'ter', 'wer', 'bleu1', 'loss'],
    'corpus_scores_bleu': ['bleu', 'ter', 'wer', 'bleu1', 'loss'],
    'corpus_scores_ter': ['ter', 'bleu', 'wer', 'bleu1', 'loss'],
    'corpus_scores_wer': ['wer', 'bleu', 'ter', 'bleu1', 'loss']
}
