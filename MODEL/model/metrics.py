import editdistance
import re

_word_re = re.compile(r"[\w]+")


def character_error_rate(y_true, y_pred, corpus_level=False):
    assert len(y_true) == len(y_pred)
    total_ed = 0
    total_chars = 0
    total_per_sample = 0.0
    n = len(y_true)
    for gt, pred in zip(y_true, y_pred):
        ed = editdistance.eval(gt, pred)
        total_ed += ed
        total_chars += len(gt)
        if len(gt) == 0:
            per_sample = 0.0
        else:
            per_sample = ed / len(gt)
        total_per_sample += per_sample
    if corpus_level:
        if total_chars == 0:
            return 0.0
        return total_ed / total_chars
    else:
        return total_per_sample / n


def word_error_rate(y_true, y_pred, corpus_level=False, pattern=_word_re):
    assert len(y_true) == len(y_pred)
    total_ed = 0
    total_words = 0
    total_per_sample = 0.0
    n = len(y_true)
    for gt, pred in zip(y_true, y_pred):
        words_gt = pattern.findall(gt)
        words_pred = pattern.findall(pred)
        if len(words_gt) == 0:
            per_sample = 0.0
            total_per_sample += per_sample
            continue
        uniq = list(set(words_gt + words_pred))
        w2i = {w: i for i, w in enumerate(uniq)}
        idx_gt = [w2i[w] for w in words_gt]
        idx_pred = [w2i[w] for w in words_pred]
        ed = editdistance.eval(idx_gt, idx_pred)
        total_ed += ed
        total_words += len(idx_gt)
        total_per_sample += ed / len(idx_gt)
    if corpus_level:
        return (total_ed / total_words) if total_words > 0 else 0.0
    else:
        return total_per_sample / n
