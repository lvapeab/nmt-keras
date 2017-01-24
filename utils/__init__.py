import callbacks
import evaluation
import numpy as np


def parse_input(line, dataset, idx2word):
    seqin = line.split()
    seqlen = len(seqin)
    seq = np.zeros(seqlen+1, dtype='int64')
    seq_words = []
    for idx, sx in enumerate(seqin):
        if idx2word.get(sx) is not None:
            seq[idx] = idx2word.get(sx)
            seq_words.append(sx)
        else:
            seq[idx] = dataset.extra_words['<unk>']
            seq_words.append('<unk>')

    seq[-1] = dataset.extra_words['<pad>']
    seq_words.append('<pad>')
    return seq, seq_words