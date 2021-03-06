from __future__ import absolute_import, division, print_function
from typing import Iterator, Mapping, Sequence, Text, Tuple

import json
import re
import tensorflow as tf

def find_lcsubstr(s1, s2):
    m = [[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p].strip(), mmax


def find_phrase_idx(text, target, phrase, last_match=False):
    lstKey = []
    lengthKey = 0
    text_orig = text
    text = text.split(" [CI] ")[0]

    countStr = text.count(phrase)
    if last_match:
        if countStr < 1:
            lstKey.append(-1)
        elif countStr == 1:
            indexKey = text.find(phrase)
            lstKey.append(indexKey)
        else:
            indexKey = text.find(phrase)
            lstKey.append(indexKey)
            while countStr > 1:
                str_new = text[indexKey+1:len(text)+1]
                indexKey_new = str_new.find(phrase)
                indexKey = indexKey+1 +indexKey_new
                lstKey.append(indexKey)
                countStr -= 1
    else:
        lstKey = (text.find(phrase),)
    return lstKey[-1], phrase


def sanitize_str(x):
    return re.sub('\s\s+', ' ', x)


def yield_sources_and_targets(input_file):
    """Reads and yields source lists and targets from the input file.

    Args:
    input_file: Path to the input file.
    input_format: Format of the input file.

    Yields:
    Tuple with (list of source texts, target text).
    """
    for sources, target in _yield_wikisplit_examples(input_file):
        yield sources, target


def filter_sources_and_targets(input_file, keys):
    for i, (sources, target) in enumerate(_yield_wikisplit_examples(input_file)):
        if keys is None or i in keys:
            yield i, sources[0], target


def _yield_wikisplit_examples(input_file):
    # The Wikisplit format expects a TSV file with the source on the first and the
    # target on the second column.
    with tf.io.gfile.GFile(input_file) as f:
        for line in f:
            line = line.replace('"', '')
            source, target = line.rstrip('\n').split('\t')
            yield [sanitize_str(source)], sanitize_str(target)


def read_label_map(path):
    """Returns label map read from the given path."""
    with tf.io.gfile.GFile(path) as f:
        if path.endswith('.json'):
            return json.load(f)
        else:
            label_map = {}
            for tag in f:
                tag = tag.strip()
                if tag:
                    label_map[tag] = len(label_map)
                else:
                    raise ValueError(
                            'There should be no empty lines in the middle of the label map '
                            'file.'
                            )
            return label_map
