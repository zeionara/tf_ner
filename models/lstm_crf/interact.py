"""Interact with a model"""
import argparse, sys
sys.path.append("../..")

from file_operations import read, write_lines

__author__ = "Guillaume Genthial"

from pathlib import Path
import functools
import json

import tensorflow as tf

from main import model_fn

#LINE = 'Президент Российской Федерации Владимир Путин'
PARAMS = './results/params.json'
MODELDIR = './results/model'


def pretty_print(line, preds):
    words = line.strip().split()
    lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
    lines = []
    for word, pred in zip(words, preds):
        line = f'{word} {pred.decode()}'
        lines.append(line)
        print(line)
    return lines
    #padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
    #padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
    #print('words: {}'.format(' '.join(padded_words)))
    #print('preds: {}'.format(' '.join(padded_preds)))


def predict_input_fn(line):
    # Words
    words = [w.encode() for w in line.strip().split()]
    nwords = len(words)

    # Wrapping in Tensors
    words = tf.constant([words], dtype=tf.string)
    nwords = tf.constant([nwords], dtype=tf.int32)

    return (words, nwords), None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--training_data', type=str, default='../../data/conll2003ru')
    parser.add_argument('--text', type=str, default='../../raw.txt')
    parser.add_argument('--output', type=str, default='../../raw.lstm-crf.predictions.txt')

    args = parser.parse_args()

    with Path(PARAMS).open() as f:
        params = json.load(f)

    params['words'] = str(Path(args.training_data, 'vocab.words.txt'))
    params['chars'] = str(Path(args.training_data, 'vocab.chars.txt'))
    params['tags'] = str(Path(args.training_data, 'vocab.tags.txt'))
    params['glove'] = str(Path(args.training_data, 'glove.npz'))

    text = read(args.text)

    estimator = tf.estimator.Estimator(model_fn, MODELDIR, params=params)
    predict_inpf = functools.partial(predict_input_fn, text)
    for pred in estimator.predict(predict_inpf):
        #print(pred.keys())
        lines = pretty_print(text, pred['tags'])
        write_lines(args.output, lines)
        break
