from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Iterator, Mapping, Sequence, Text, Tuple

import json
import re
import tensorflow as tf

def find_lcsubstr(s1, s2):
  m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]
  mmax=0
  p=0
  for i in range(len(s1)):
    for j in range(len(s2)):
      if s1[i]==s2[j]:
        m[i+1][j+1]=m[i][j]+1
        if m[i+1][j+1]>mmax:
          mmax=m[i+1][j+1]
          p=i+1
  return s1[p-mmax:p].strip(), mmax


def remove_punct(phrase, target):
  phrase = re.sub('\s[?!,]$', '', phrase.strip().lstrip(' ,'))
  if phrase.endswith(' .'):
    target = ' '.join(target.strip().split())
    # Make sure phrase is at end of target string, else this is abbreviation
    if target.find(phrase) + len(phrase) == len(target):
      phrase = phrase.rstrip(' .')
  return phrase


def find_phrase_idx(text, target, phrase, last_match=False, stop_phrs=None):
  lstKey = []
  lengthKey = 0
  text_orig = text
  text = text.split(" [CI] ")[0]
  ignore_phr = False
  if text.find(phrase) == -1:
    phrase_spl = tuple(remove_punct(phrase, target).split())
    for phr in stop_phrs:
      m = len(phr)
      if phrase_spl == phr:
        ignore_phr = True
        break
      if phrase_spl[:m] == phr and " ".join(phrase_spl[m:]) in text:
        phrase = " ".join(phrase_spl[m:])
        break
      if phrase_spl[-m:] == phr and " ".join(phrase_spl[:-m]) in text:
        phrase = " ".join(phrase_spl[:-m])
        break

  #if text.find(phrase) == -1:
  #  sub_phrase, _ = find_lcsubstr(text, phrase)
  #  if sub_phrase != '':
  #     phrase = sub_phrase

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
  return lstKey[-1], phrase, ignore_phr

def get_token_list(text):
  """Returns a list of tokens.

  This function expects that the tokens in the text are separated by space
  character(s). Example: "ca n't , touch". This is the case at least for the
  public DiscoFuse and WikiSplit datasets.

  Args:
    text: String to be split into tokens.
  """
  return text.split()


def yield_sources_and_targets(
    input_file,
    input_format):
  """Reads and yields source lists and targets from the input file.

  Args:
    input_file: Path to the input file.
    input_format: Format of the input file.

  Yields:
    Tuple with (list of source texts, target text).
  """
  if input_format == 'wikisplit':
    yield_example_fn = _yield_wikisplit_examples
  elif input_format == 'discofuse':
    yield_example_fn = _yield_discofuse_examples
  elif input_format == 'rewrite':
    yield_example_fn = _yield_rewrite_examples
  else:
    raise ValueError('Unsupported input_format: {}'.format(input_format))

  for sources, target in yield_example_fn(input_file):
    yield sources, target


def _yield_wikisplit_examples(
    input_file):
  # The Wikisplit format expects a TSV file with the source on the first and the
  # target on the second column.
  with tf.io.gfile.GFile(input_file) as f:
    for line in f:
      line = line.replace('"', '')
      source, target = line.rstrip('\n').split('\t')
      yield [source], target


def _yield_discofuse_examples(
    input_file):
  """Yields DiscoFuse examples.

  The documentation for this format:
  https://github.com/google-research-datasets/discofuse#data-format

  Args:
    input_file: Path to the input file.
  """
  with tf.io.gfile.GFile(input_file) as f:
    for i, line in enumerate(f):
      if i == 0:  # Skip the header line.
        continue
      coherent_1, coherent_2, incoherent_1, incoherent_2, _, _, _, _ = (
          line.rstrip('\n').split('\t'))
      # Strip because the second coherent sentence might be empty.
      fusion = (coherent_1 + ' ' + coherent_2).strip()
      yield [incoherent_1, incoherent_2], fusion

def _yield_rewrite_examples(input_file):
  with tf.io.gfile.GFile(input_file) as f:
    for line in f:
      line = line.replace('"', '')
      text = line.rstrip('\n').split('\t')
      context, cur_utt, target = text[0], text[1], text[2]
      yield [context, cur_utt], target


def read_label_map(path):
  """Returns label map read from the given path."""
  with tf.io.gfile.GFile(path) as f:
    if path.endswith('.json'):
      return json.load(f)
    else:
      label_map = {}
      empty_line_encountered = False
      for tag in f:
        tag = tag.strip()
        if tag:
          label_map[tag] = len(label_map)
        else:
          if empty_line_encountered:
            raise ValueError(
                'There should be no empty lines in the middle of the label map '
                'file.'
                )
            empty_line_encountered = True
      return label_map