# -*- coding: utf-8 -*-

from __future__ import division
import theano
import theano.tensor as T
import numpy
import logging
from itertools import izip
import time
import sys
import subprocess
import re

def _log(s, nl=True):
    s = str(s)
    if nl:
        sys.stderr.write('{}\n'.format(s))
    else:
        sys.stderr.write('{}'.format(s))
    sys.stderr.flush()

DEBUG = True
def debug(s, nl=True):
    if DEBUG:
        if nl:
            sys.stderr.write('{}\n'.format(s))
        else:
            sys.stderr.write('{}'.format(s))
        sys.stderr.flush()

# exeTime
def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        sys.stderr.write('@{}, {} start\n'.format(time.strftime(
            "%X", time.localtime()), func.__name__))
        # print "@%s, {%s} start" % (time.strftime("%X", time.localtime()),
        # func.__name__)
        back = func(*args, **args2)
        sys.stderr.write('@{}, {} end\n'.format(time.strftime(
            "%X", time.localtime()), func.__name__))
        # print "@%s, {%s} end" % (time.strftime("%X", time.localtime()),
        # func.__name__)
        sys.stderr.write('@{}s taken for {}\n'.format(
            format(time.time() - t0, '0.3f'), func.__name__))
        # print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back
    return newFunc


class param_init(object):

    def __init__(self, **kwargs):

        self.shared = kwargs.pop('shared', True)

    def param(self, size, init_type=None, name=None, **kwargs):
        try:
            if init_type is not None:
                func = getattr(self, init_type)
            elif len(size) == 1:
                func = getattr(self, 'constant')
            elif size[0] == size[1]:
                func = getattr(self, 'orth')
            else:
                func = getattr(self, 'normal')
        except AttributeError:
            sys.stderr.write('AttributeError, {}'.format(init_type))
        else:
            param = func(size, **kwargs)
        if self.shared:
            param = theano.shared(value=param, borrow=True, name=name)
        return param

    def uniform(self, size, **kwargs):
        # low = kwargs.pop('low', -6./sum(size))
        # high = kwargs.pop('high', 6./sum(size))
        low = kwargs.pop('low', -0.01)
        high = kwargs.pop('high', 0.01)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.uniform(low=low, high=high, size=size),
            dtype=theano.config.floatX)
        return param

    def normal(self, size, **kwargs):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 0.01)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.normal(loc=loc, scale=scale, size=size),
            dtype=theano.config.floatX)
        return param

    def constant(self, size, **kwargs):
        value = kwargs.pop('scale', 0.)
        param = numpy.ones(size, dtype=theano.config.floatX) * value
        return param

    def orth(self, size, **kwargs):
        scale = kwargs.pop('scale', 1.0)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if len(size) != 2:
            raise ValueError
        if size[0] == size[1]:
            M = rng.randn(*size).astype(theano.config.floatX)
            Q, R = numpy.linalg.qr(M)
            Q = Q * numpy.sign(numpy.diag(R))
            param = Q * scale
            return param
        else:
            M1 = rng.randn(size[0], size[0]).astype(theano.config.floatX)
            M2 = rng.randn(size[1], size[1]).astype(theano.config.floatX)
            Q1, R1 = numpy.linalg.qr(M1)
            Q2, R2 = numpy.linalg.qr(M2)
            Q1 = Q1 * numpy.sign(numpy.diag(R1))
            Q2 = Q2 * numpy.sign(numpy.diag(R2))
            n_min = min(size[0], size[1])
            param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            return param

    def mfunc(self, size, m=3, **kwargs):
        if size[0] == size[1]:
            func = self.orth
        else:
            func = self.normal
        params = [func(size) for _ in range(m)]
        return numpy.concatenate(params, axis=1)


'''
# layer normalization
def ln(input, g, b):
    _eps = 1e-5
    _mu = T.mean(input, axis=1, keepdims=True)
    _sigma = T.var(input, axis=1, keepdims=True)
    output = (input - _mu) / T.sqrt(_sigma + _eps)
    output = g[None, :] * output + b[None, :]
    return output
'''

# layer normalization


def ln(x, s, b):
    _eps = 1e-5
    output = (x - x.mean(1)[:, None]) / theano.tensor.sqrt((x.var(1)[:, None] + _eps))
    output = s[None, :] * output + b[None, :]
    return output


def repeat_x(x, n_times):
    # This is black magic based on broadcasting,
    # that's why variable names don't make any sense.
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    out = a * b
    return out


def adadelta(parameters, gradients, rho=0.95, eps=1e-6):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32'))
                    for p in parameters]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32'))
                 for p in parameters]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [rho * g_sq + (1 - rho) * (g**2)
                        for g_sq, g in izip(gradients_sq, gradients)]

    # calculates the step in direction. The square root is an approximation to
    # getting the RMS for the average value
    deltas = [(T.sqrt(d_sq + eps) / T.sqrt(g_sq + eps)) * grad for d_sq,
              g_sq, grad in izip(deltas_sq, gradients_sq_new, gradients)]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [rho * d_sq + (1 - rho) * (d**2)
                     for d_sq, d in izip(deltas_sq, deltas)]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq, gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq, deltas_sq_new)
    # parameters_updates = [ (p,T.clip(p - d, -15, 15)) for p,d in izip(parameters,deltas) ]
    parameters_updates = [(p, (p - d)) for p, d in izip(parameters, deltas)]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


def step_clipping(params, gparams, scale=1.):
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
    notfinite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    multiplier = T.switch(grad_norm < scale, 1., scale / grad_norm)
    _g = []
    for param, gparam in izip(params, gparams):
        tmp_g = gparam * multiplier
        _g.append(T.switch(notfinite, param * 0.1, tmp_g))

    params_clipping = _g

    return params_clipping


import os
import shutil


def init_dir(dir_name, delete=False):

    if not dir_name == '':
        if os.path.exists(dir_name):
            if delete:
                shutil.rmtree(dir_name)
                _log('{} exists, delete'.format(dir_name))
            else:
                _log('{} exists, no delete'.format(dir_name))
        else:
            os.mkdir(dir_name)
            _log('Create {}\n'.format(dir_name))


def _index2sentence(vec, dic):
    r = [dic[index] for index in vec]
    return " ".join(r)


def _filter_reidx(bos_id, eos_id, best_trans, tV_i2w=None, ifmv=False, ptv=None):
    if ifmv and ptv is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        true_idx = [ptv[i] for i in best_trans]
    else:
        true_idx = best_trans
    true_idx = filter(lambda y: y != eos_id and y != bos_id, true_idx)
    best_trans = _index2sentence(true_idx, tV_i2w)
    return best_trans


def part_sort(vec, num):
    '''
    vec:    [ 3,  4,  5, 12,  1,  3,  29999, 33,  2, 11,  0]
    '''

    idx = numpy.argpartition(vec, num)[:num]

    '''
    put k-min numbers before the _th position and get indexes of the k-min numbers in vec (unsorted)
    idx = np.argpartition(vec, 5)[:5]:
        [ 4, 10,  8,  0,  5]
    '''

    kmin_vals = vec[idx]

    '''
    kmin_vals:  [1, 0, 2, 3, 3]
    '''

    k_rank_ids = numpy.argsort(kmin_vals)

    '''
    k_rank_ids:    [1, 0, 2, 3, 4]
    '''

    k_rank_ids_invec = idx[k_rank_ids]

    '''
    k_rank_ids_invec:  [10,  4,  8,  0,  5]
    '''

    '''
    sorted_kmin = vec[k_rank_ids_invec]
    sorted_kmin:    [0, 1, 2, 3, 3]
    '''

    return k_rank_ids_invec


def part_sort_lg2sm(vec, klargest):
    # numpy.argpartition(vec, num)
    sz = len(vec)
    left = sz - klargest
    ind = numpy.argpartition(vec, left)[left:]
    # sort the small data before the num_th element, and get the index
    # index of k largest elements (lg to sm)
    return ind[numpy.argsort(-vec[ind])]


def euclidean(n1, n2):
    # numpy.float64 (0. ~ inf)
    return numpy.linalg.norm(n1 - n2)


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_better(x):
    return (numpy.arctan(x / 100) / numpy.pi) + 0.5


def logistic(x):
    return 1 / (1 + numpy.exp(-x / 10000.))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()


def kl_dist(p, q):
    p = numpy.asarray(p, dtype=numpy.float)
    q = numpy.asarray(q, dtype=numpy.float)
    return numpy.sum(numpy.where(p != 0, (p - q) * numpy.log10(p / q), 0))


def back_tracking(beam, best_sample_endswith_eos, detail=True):
    # (0.76025655120611191, [29999], 0, 7)
    if len(best_sample_endswith_eos) == 5:
        best_loss, accum, w, bp, endi = best_sample_endswith_eos
    else:
        best_loss, w, bp, endi = best_sample_endswith_eos
    # from previous beam of eos beam, firstly bp:j is the item index of
    # {end-1}_{th} beam
    seq = []
    for i in reversed(xrange(1, endi)):
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # contain eos last word, finally we filter, so no matter
        if detail:
            _, _, _, _, w, backptr = beam[i][bp]
        else:
            _, _, w, _, backptr = beam[i][bp]
        #_, _, _, w, backptr = beam[i][bp]
        seq.append(w)
        bp = backptr
    return seq[::-1], best_loss  # reverse


def init_beam(beam, cnt=50, init_score=0.0, init_loss=0.0, init_state=None, detail=True, cp=False):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)
    # (sum score i, state i, yi, backptr), indicator for the first target word (bos <S>)
    if cp is True:
        init_y_emb_im1, init_hi, init_pi, init_ai, init_si, init_moi, init_sci, init_cei = \
                None, None, None, None, None, None, None, None
        beam[0].append((init_score, init_loss, init_y_emb_im1, init_hi, init_pi, init_ai,
                        init_si, init_moi, init_sci, init_cei, -1, 0))
    elif detail is True:
        init_pi = None
        beam[0].append((init_score, init_loss, init_pi, init_state, -1, 0))
    else:
        beam[0].append((init_score, init_state, -1, 0))
    #beam[0].append((init_score, init_pi, init_state, -1, 0))
    # such as: beam[0] is (0.0 (-log(1)), 'Could', 0)

def init_beam_sm(beam, cnt=50, init_score=0.0, init_state=None, init_y_emb_im1=None):
    del beam[:]
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)
    # (sum score i, y_emb_im1, state i, yi, backptr), indicator for the first target word (bos <S>)
    beam[0].append((init_score, init_state, -1, init_y_emb_im1, 0))

def dec_conf(switchs, k, mode, kl, nprocess, lm, ngram, alpha, beta, valid_set):
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, ifwatch_adist, merge_way, \
        ifapprox_dist, ifapprox_att, ifadd_lmscore, ifsplit = switchs
    sys.stderr.write(
        '\n.........................decoder config..........................\n')
    if mode == 0:
        sys.stderr.write('# MLE search => ')
    elif mode == 1:
        sys.stderr.write('# Original beam search => ')
    elif mode == 2:
        sys.stderr.write('# Naive beam search => ')
    elif mode == 3:
        sys.stderr.write('# Cube pruning => ')

    sys.stderr.write('\n\t Beam size : {}'
                     '\n\t KL distance threshold : {}'
                     '\n\t Processes number : {}'
                     '\n\t Decoding validation set ? {}'
                     '\n\t Using batch ? {}'
                     '\n\t Using softmax ? {}'
                     '\n\t Using normalized ? {}'
                     '\n\t Manipulating vocab : {}'
                     '\n\t Watching attention distribution ? {}'
                     '\n\t Cube pruning merge way : {}'
                     '\n\t totally approximating ? {}'
                     '\n\t Only using approximate attention ? {}'
                     '\n\t Consider lm score ? {}'
                     '\n\t Split fn_next ? {}'
                     '\n\t Validation set file : {}'
                     '\n\t Language model file : {}'
                     '\n\t Ngrams : {}'
                     '\n\t Length normalization : {}'
                     '\n\t Coverage penalty : {}\n\n'.format(
                         k,
                         kl,
                         nprocess,
                         True if ifvalid else False,
                         True if ifbatch else False,
                         False if ifscore else True,
                         True if ifnorm else False,
                         True if ifmv else False,
                         True if ifwatch_adist else False,
                         merge_way,
                         True if ifapprox_dist else False,
                         True if ifapprox_att else False,
                         True if ifadd_lmscore else False,
                         True if ifsplit else False,
                         valid_set if ifvalid else 'None',
                         lm,
                         ngram,
                         alpha,
                         beta)
                     )
    sys.stdout.flush()


def append_file(file_prefix, content):
    f = open(file_prefix, 'a')
    f.write(content)
    f.write('\n')
    f.close()


def fetch_bleu_from_file(fbleufrom):
    fread = open(fbleufrom, 'r')
    result = fread.readlines()
    fread.close()
    f_bleu = 0.
    f_multibleu = 0.
    for line in result:
        bleu_pattern = re.search(r'BLEU score = (0\.\d+)', line)
        if bleu_pattern:
            s_bleu = bleu_pattern.group(1)
            f_bleu = format(float(s_bleu) * 100, '0.2f')
        multi_bleu_pattern = re.search(r'BLEU = (\d+\.\d+)', line)
        if multi_bleu_pattern:
            s_multibleu = multi_bleu_pattern.group(1)
            f_multibleu = format(float(s_multibleu), '0.2f')
    return f_bleu, f_multibleu


# valid_out: valids/trans...
def valid_bleu(valid_out, val_tst_dir, val_prefix):

    save_log = '{}.{}'.format(valid_out, 'log')

    cmd = ['sh evaluate.sh {} {} {} {}'.format(
        valid_out,
        val_prefix,
        save_log,
        val_tst_dir)
    ]

    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)

    bleu_out = child.communicate()
    child.wait()
    mteval_bleu, multi_bleu = fetch_bleu_from_file(save_log)
    os.remove(save_log)
    # we use the bleu without unk and mteval-v11, process reference
    return mteval_bleu, multi_bleu
