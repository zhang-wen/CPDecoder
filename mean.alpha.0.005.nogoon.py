from __future__ import division

import theano
import theano.tensor as T
import os
import sys

from utils import adadelta, step_clipping, init_dir, log
from stream_with_dict import get_tr_stream, ensure_special_tokens
import configurations
from cp_sample import Translator
import cPickle as pickle

from trans_model import Translate
import subprocess

import numpy as np
import time
import collections

if __name__ == "__main__":

    config = getattr(configurations, 'get_config_cs2en')()

    log('\nLoad source and target vocabulary ...')
    n_src_words = config['src_vocab_size']
    n_trg_words = config['trg_vocab_size']
    log('Want to generate source dict {} and target dict {}'.format(
        n_src_words, n_trg_words))
    sv = pickle.load(open(config['src_vocab']))
    tv = pickle.load(open(config['trg_vocab']))
    log('Source vocab count: {}, target vocab count: {}'.format(len(sv), len(tv)))
    log('Vocabulary contains <S>, <UNK> and </S>')

    seos_idx, teos_idx = n_src_words - 1, n_trg_words - 1
    sv = ensure_special_tokens(
        sv, bos_idx=0, eos_idx=seos_idx, unk_idx=config['unk_id'])
    tv = ensure_special_tokens(
        tv, bos_idx=0, eos_idx=teos_idx, unk_idx=config['unk_id'])

    # the tv is originally:
    #   {'UNK': 1, '<s>': 0, '</s>': 0, 'is': 5, ...}
    # after ensure_special_tokens, the tv becomes:
    #   {'<UNK>': 1, '<S>': 0, '</S>': trg_vocab_size-1, 'is': 5, ...}
    tv_i2w = {i: w for w, i in tv.iteritems()}
    sv_i2w = {i: w for w, i in sv.iteritems()}
    # after reversing, the tv_i2w become:
    #   {1: '<UNK>', 0: '<S>', trg_vocab_size-1: '</S>', 5: 'is', ...}

    init_dir(config['models_dir'])
    init_dir(config['val_out_dir'])
    init_dir(config['tst_out_dir'])

    source = T.lmatrix('source')
    target = T.lmatrix('target')
    source_mask = T.matrix('source_mask')
    target_mask = T.matrix('target_mask')
    # for each batch which is a data in tr_stream.get_epoch_iterator(),
    # we set the maximum sentence length in this batch as sent_len;
    # source, source_mask, target and target_mask are all matrix shape: (batch_size * sent_len)
    # and their type are all theano.tensor.var.TensorVariable

    ltopk_trg_vocab_idx = []
    if config['use_mv']:
        # no need to use the whole vocabulary
        v_part = T.vector('batch_target_vocab', dtype='int64')
        v_true = T.matrix('v_true', dtype='int64')

        from manvocab import topk_target_vcab_list
        ltopk_trg_vocab_idx = topk_target_vcab_list(**config)
        log('{}'.format(ltopk_trg_vocab_idx))
        log('{}'.format([tv_i2w[i] for i in ltopk_trg_vocab_idx]))

    one_model = config['one_model']
    log('Build lookup table, bi-directional encoder and decoder ... ', nl=False)

    trans = Translate(**config)
    # transpose all the input matrix into shape (sent_len * batch_size)
    if config['use_mv']:
        trans.apply(source.T, source_mask.T, target.T,
                    target_mask.T, v_part, v_true)
    else:
        trans.apply(source.T, source_mask.T, target.T, target_mask.T)
    log('Done\n')

    if config['reload']:
        log('Reload model {}'.format(config['one_model']))
        trans.load(one_model)

    # actually the average cross entropy (cost) per sentence in a batch
    cost = trans.mean_cost
    log_norm = trans.mean_abs_log_norm
    params = trans.params

    # print all parameters in this rnn search
    for value in params:
        log('\t{:15}: {}'.format(value.get_value().shape, value.name))

    log('Build grad ... ', nl=False)
    grade = T.grad(cost, params)
    # add step clipping, L2-norm of grade to prevent over-fitting, make
    # gradients (update) smaller, model simpler
    if config['step_clipping'] > 0.:
        grade = step_clipping(params, grade, config['step_clipping'])
    updates = adadelta(params, grade)
    log('Done')

    log('Build translation model tr_fn ... ', nl=False)
    # by using adadelta, we update parameters, gradients and deltax
    if config['use_mv']:
        inps = [source, source_mask, target, target_mask, v_part, v_true]
    else:
        inps = [source, source_mask, target, target_mask]
    tr_fn = theano.function(inps, [cost, log_norm], updates=updates)
    log('Done')

    log('Build sample model f_init f_nh f_na f_ns f_mo f_ws f_ps f_p ... ', nl=False)
    fs = trans.build_sample()
    log('Done')

    k_batch_start_sample = config['k_batch_start_sample']
    batch_size, sample_size = config['batch_size'], config['hook_samples']
    if batch_size < sample_size:
        log('Batch size must be great or equal with sample size')
        sys.exit(0)

    batch_start_sample = np.random.randint(
        2, k_batch_start_sample)  # [low, high)
    log('will randomly generate {} sample at {}th batch'.format(
        sample_size, batch_start_sample))

    batch_count, sent_count, val_time, best_score = 0, 0, 0, 0.
    model_name = ''
    sample_src_np, sample_trg_np = None, None

    switchs = [0, config['use_batch'], config['use_score'], config['use_norm'], config[
        'use_mv'], config['watch_adist'], config['merge_way'], config['avg_att']]
    beam_size = config['beam_size']
    search_mode = config['search_mode']
    lmpath = config['lm_path'] if config['lm_path'] is not None else None
    lm = kenlm.Model(lmpath) if (lmpath and search_mode == 3) else None

    translator = Translator(
        fs=fs,
        switchs=switchs,
        mode=search_mode,
        svcb=sv,
        svcb_i2w=sv_i2w,
        tvcb=tv,
        tvcb_i2w=tv_i2w,
        ngram=config['ngram'],
        k=beam_size,
        thresh=config['m_threshold'],
        lm=lm,
        ln_alpha=config['length_norm'],
        cp_beta=config['cover_penalty']
    )

    start_time = time.time()
    tr_stream = get_tr_stream(**config)
    log('Start training!!!')
    max_epochs = config['max_epoch']
    allv = []
    npv = None
    fix_npv = None
    fix_npv_true = None
    for epoch in range(max_epochs):
        # take the batch sizes 3 as an example:
        # tuple: tuple[0] is indexes of source sentence (np.ndarray)
            # like array([[0, 23, 3, 4, 29999], [0, 2, 1, 29999], [0, 31, 333, 2, 1, 29999]])
        # tuple: tuple[1] is indexes of source sentence mask (np.ndarray)
            # like array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        # tuple: tuple[2] is indexes of target sentence (np.ndarray)
        # tuple: tuple[3] is indexes of target sentence mask (np.ndarray)
        # tuple: tuple[4] is dict [0, 3, 4, 2, 29999]   # no duplicated word
        # their shape: (batch_size * sentence_length)
        epoch_start = time.time()
        eidx = epoch + 1
        log('....................... Epoch [{} / {}] .......................'.format(
            eidx, max_epochs)
            )
        n_samples = 0
        batch_count_in_cur_epoch = 0
        tr_epoch_mean_cost = 0.
        for tr_data in tr_stream.get_epoch_iterator():  # tr_data is a tuple  update one time for one batch

            batch_count += 1
            batch_count_in_cur_epoch += 1

            bx, bxm, by, bym, btvob = tr_data[0], tr_data[
                1], tr_data[2], tr_data[3], tr_data[4]
            minibatch_size = by.shape[0]
            y_maxlen = by.shape[1]
            n_samples += minibatch_size
            ud_start = time.time()

            if config['use_mv']:
                del allv[:]
                allv.extend(ltopk_trg_vocab_idx)
                for v in btvob:
                    allv.extend(v)
                # batch level voc.
                allv = sorted(set([v for v in allv if v < n_trg_words]))
                npv = np.zeros(len(allv)).astype('int64')
                for i, vid in enumerate(allv):
                    npv[i] = vid
                # generate the y index in batch level voc.
                npv_true = np.zeros((y_maxlen, minibatch_size)).astype('int64')
                for sid, y in enumerate(by):
                    y[np.where(y >= n_trg_words)] = 1
                    for idx, yidx in enumerate(y):
                        npv_true[idx, sid] = allv.index(yidx)
                # cost = tr_fn(*tr_data)  # <type 'np.ndarray'>
                # array(232.33)
                cost, log_norm = tr_fn(bx, bxm, by, bym, npv, npv_true)
            else:
                cost, log_norm = tr_fn(bx, bxm, by, bym)
            # array(232.33)
            ud = time.time() - ud_start
            tr_epoch_mean_cost += float(cost)

            if batch_count % config['display_freq'] == 0:

                runtime = (time.time() - start_time) / 60.
                ref_wcnt_wopad = np.count_nonzero(bym)
                ws_per_sent = ref_wcnt_wopad / minibatch_size
                sec_per_sent = ud / minibatch_size

                log(
                    '[e {:>2}]  '
                    '[b {: >4}]  '
                    '[samples {: >7}]  '
                    '[loss=>{: >8}]  '
                    '[words/s=>{: >4}/{: >2}={:>6}]  '
                    '[upd/s=>{:>6}/{: >2}={: >5}s]  '
                    '[vcb {: >4}]  '
                    '[logZ {: >4}] '
                    '[elapsed {:.3f}m]'.format(
                        eidx,
                        batch_count_in_cur_epoch,
                        n_samples,
                        format(float(cost), '0.3f'),
                        ref_wcnt_wopad, minibatch_size, format(
                            ws_per_sent, '0.3f'),
                        format(ud, '0.3f'), minibatch_size, format(
                            sec_per_sent, '0.3f'),
                        len(allv),
                        format(float(log_norm), '0.3f'),
                        runtime)
                )

            if batch_count % config['sampling_freq'] == 0:

                if sample_src_np is not None:
                    t = Translator(
                        fs=fs,
                        switchs=switchs,
                        mode=search_mode,
                        svcb=sv,
                        svcb_i2w=sv_i2w,
                        tvcb=tv,
                        tvcb_i2w=tv_i2w,
                        ngram=config['ngram'],
                        k=beam_size,
                        thresh=config['m_threshold'],
                        lm=lm,
                        ptv=npv,
                        ln_alpha=config['length_norm'],
                        cp_beta=config['cover_penalty']
                    )

                    t.trans_samples(sample_src_np, sample_trg_np)
                else:
                    t = Translator(
                        fs=fs,
                        switchs=switchs,
                        mode=search_mode,
                        svcb=sv,
                        svcb_i2w=sv_i2w,
                        tvcb=tv,
                        tvcb_i2w=tv_i2w,
                        ngram=config['ngram'],
                        k=beam_size,
                        thresh=config['m_threshold'],
                        lm=lm,
                        ptv=fix_npv,
                        ln_alpha=config['length_norm'],
                        cp_beta=config['cover_penalty']
                    )

                    t.trans_samples(bx[:sample_size], by[:sample_size])

            # sample, just take a look at the translate of some source
            # sentences in training data
            if config['if_fixed_sampling'] and batch_count == batch_start_sample:
                # select k sample from current batch
                # rand_rows = random.sample(xrange(batch_size), sample_size)
                rand_rows = np.random.choice(
                    batch_size, sample_size, replace=False)
                # randomly select sample_size number from batch_size
                # rand_rows = np.random.randint(batch_size, size=sample_size)   #
                # np.int64, may repeat
                sample_src_np = np.zeros(
                    shape=(sample_size, bx.shape[1])).astype('int64')
                sample_trg_np = np.zeros(
                    shape=(sample_size, by.shape[1])).astype('int64')
                for id in xrange(sample_size):
                    sample_src_np[id, :] = bx[rand_rows[id], :]
                    sample_trg_np[id, :] = by[rand_rows[id], :]

                if config['use_mv']:
                    fix_npv = npv
                    fix_npv_true = npv_true

            if config['epoch_eval'] is not True and batch_count > config['val_burn_in'] and \
               batch_count % config['bleu_val_freq'] == 0:

                # translate dev
                val_time += 1
                log('Batch [{}], valid time [{}], save model ...'.format(
                    batch_count, val_time))

                # save models: search_model_ch2en/params_e5_upd3000.npz
                if config['save_one_model']:
                    model_name = '{}.{}'.format(config['model_prefix'], 'npz')
                else:
                    model_name = '{}_e{}_upd{}.{}'.format(
                        config['model_prefix'], eidx, batch_count, 'npz')

                trans.savez(model_name)
                cmd = ['sh trans.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                    eidx,
                    batch_count,
                    model_name,
                    search_mode,
                    beam_size,
                    config['use_norm'],
                    config['use_batch'],
                    config['use_score'],
                    1,
                    config['use_mv'],
                    config['merge_way'],
                    config['m_threshold'],
                    config['ngram'],
                    config['length_norm'],
                    config['cover_penalty'],
                    config['val_out_dir'])
                ]

                child = subprocess.Popen(cmd, shell=True)

        mean_cost_on_tr_data = tr_epoch_mean_cost / batch_count_in_cur_epoch
        epoch_time_consume = time.time() - epoch_start
        log('End epoch [{}], average cost on all training data: {}, consumes time:'
            '{}s'.format(eidx, mean_cost_on_tr_data, format(epoch_time_consume, '0.3f')))

        if config['epoch_eval']:
            # translate dev
            val_time += 1
            log('Batch [{}], valid time [{}], save model ...'.format(
                batch_count, val_time))
            # save models: search_model_ch2en/params_e5_upd3000.npz
            if config['save_one_model']:
                model_name = '{}.{}'.format(config['model_prefix'], 'npz')
            else:
                model_name = '{}_e{}_upd{}.{}'.format(
                    config['model_prefix'], eidx, batch_count, 'npz')

            trans.savez(model_name)
            log('Start decoding on validation data [{}]...'.format(
                config['val_set']))

            cmd = ['sh trans.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(
                eidx,
                batch_count,
                model_name,
                search_mode,
                beam_size,
                config['use_norm'],
                config['use_batch'],
                config['use_score'],
                1,
                config['use_mv'],
                config['merge_way'],
                config['m_threshold'],
                config['ngram'],
                config['length_norm'],
                config['cover_penalty'],
                config['val_out_dir'])
            ]

            child = subprocess.Popen(cmd, shell=True)

    tr_time_consume = time.time() - start_time
    log('Training consumes time: {}s'.format(format(tr_time_consume, '0.3f')))
