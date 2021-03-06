# -*- coding: utf-8 -*-

import os
import theano.tensor as T

from stream_with_dict import get_dev_stream, get_tst_stream, ensure_special_tokens
import configurations
from cp_sample import Translator
import cPickle as pickle

from trans_model import Translate
# Get the arguments
import sys
import subprocess
import numpy as np
import time
import argparse
from utils import dec_conf, _log, init_dir, valid_bleu, append_file
from wlm import load_language_model
import kenlm

DEBUG = True


def debug(s, nl=True):
    if DEBUG:
        if nl:
            sys.stderr.write(s + '\n')
        else:
            sys.stderr.write(s)
        sys.stderr.flush()


if __name__ == "__main__":
    decoder = argparse.ArgumentParser(prog='NMT decoder')

    decoder.add_argument(
        '--epoch',
        dest='epoch',
        type=int,
        default=0,
        help='Which epoch model is saved in.',
    )

    decoder.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=0,
        help='Which batch model is saved in.',
    )

    decoder.add_argument(
        '--model-name',
        dest='model_name',
        help='Model name.',
    )

    decoder.add_argument(
        '--search-mode',
        dest='search_mode',
        type=int,
        help='Search mode: naive:0/mle:1/beam search:2/cube pruning:3',
    )

    decoder.add_argument(
        '--beam-size',
        dest='beam_size',
        type=int,
        default=0,
        help='Beam size of beam search. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--use-valid',
        dest='use_valid',
        type=int,
        default=0,
        help='Translate valid set. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--valid-set',
        dest='valid_set',
        help='valid set: (nist02, nist03, ...)',
    )

    decoder.add_argument(
        '--use-batch',
        dest='use_batch',
        type=int,
        default=0,
        help='Whether we apply batch on beam search. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--use-score',
        dest='use_score',
        type=int,
        default=0,
        help='Whether we use model score instead of softmax prob. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--use-norm',
        dest='use_norm',
        type=int,
        default=0,
        help='Evaluate fianl score by using sentence-level normalization. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--use-mv',
        dest='use_mv',
        type=int,
        default=0,
        help='We use manipulation vacabulary by add this parameter. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--ifwatch-adist',
        dest='watch_adist',
        type=int,
        default=0,
        help='Whether we watch the distribution of attention probs. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--merge-way',
        dest='merge_way',
        default='Him1',
        help='merge way in cube pruning. (DEFAULT=s_im1. Him1/Hi/AiKL/LM)',
    )

    decoder.add_argument(
        '--ifapprox-dist',
        dest='ifapprox_dist',
        type=int,
        default=0,
        help='Whether we average whole thing in cube. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--ifapprox-att',
        dest='ifapprox_att',
        type=int,
        default=0,
        help='Whether we average attention vector. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--ifadd-lmscore',
        dest='add_lmscore',
        type=int,
        default=0,
        help='Whether we consider the lm score. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--ifsplit',
        dest='ifsplit',
        type=int,
        default=0,
        help='Whether we split the fn_next. (DEFAULT=0)',
    )

    decoder.add_argument(
        '--m-threshold',
        dest='m_threshold',
        type=float,
        default=0.,
        help='a super-parameter to merge attention vector in cube pruning. (DEFAULT=0. no merge)',
    )

    decoder.add_argument(
        '--n-process',
        dest='n_process',
        type=int,
        default=5,
        help='the processes count we use on cpu in decoding test set. (DEFAULT=5)',
    )

    decoder.add_argument(
        '--lm-path',
        dest='lm_path',
        default=None,
        help='KenLM trained language model path (arpa or binary). (DEFAULT=None)',
    )

    decoder.add_argument(
        '--ngram',
        dest='ngram',
        type=int,
        default=3,
        help='restored into prefix trie (trie[0], trie[1], ..., trie[ngram-1]), should smaller '
        'than n-gram language model. (DEFAULT=3)',
    )

    decoder.add_argument(
        '--length-norm',
        dest='length_norm',
        type=float,
        default=0.,
        help='the parameter alpha to control the strenght of the length normalization.'
        '(DEFAULT=0. no length normalization)',
    )

    decoder.add_argument(
        '--cover-penalty',
        dest='cover_penalty',
        type=float,
        default=0.,
        help='the parameter beta to control the strength of the coverage penalty.'
        '(DEFAULT=0. no coverage penalty)',
    )

    decoder.add_argument(
        '--when-train',
        dest='when_train',
        type=int,
        default=0,
        help='whether is training'
        '(DEFAULT=no)',
    )


    decoder.add_argument(
        '--workspace',
        dest='workspace',
        type=str,
        default='wvalids',
        help='valid out directory if we translate validation.'
    )

    args = decoder.parse_args()
    epoch = args.epoch
    batch = args.batch
    model_name = args.model_name
    beam_size = args.beam_size
    # print type(beam), beam  # <type 'str'> 5
    search_mode = args.search_mode
    switchs = [args.use_valid, args.use_batch, args.use_score, args.use_norm, args.use_mv,
               args.watch_adist, args.merge_way, args.ifapprox_dist, args.ifapprox_att,
               args.add_lmscore, args.ifsplit]
    valid_set = args.valid_set
    kl = args.m_threshold
    nprocess = args.n_process
    lmpath = args.lm_path if args.lm_path is not None else None
    ngram = args.ngram

    alpha = args.length_norm
    beta = args.cover_penalty

    dec_conf(switchs, beam_size, search_mode, kl, nprocess, lmpath, ngram, alpha, beta, valid_set)

    config = getattr(configurations, 'get_config_cs2en')()
    _log('init decoder ... ')
    trans = Translate(**config)
    _log('done')

    _log('build decode funcs: f_init f_nh f_na f_ns f_mo f_ws f_ps f_ce f_next f_emb ... ', nl=False)
    fs = trans.build_sample()
    _log('done')

    y_im1 = [2]
    npy = np.asarray(y_im1)
    if npy.ndim == 1:
        x = npy[None, :]
    # context = np.random.sample((7, 1, 2048)).astype(np.float32)
    # s_im1 = np.random.sample((1, 1024)).astype(np.float32)
    debug('............. time testing ..............................')
    s = time.time()
    s_im1, ctx, c_x = fs[0](x)
    e = time.time()
    tinit = (e - s)

    s = time.time()
    #yemb_im1, hi = fs[1](*[y_im1, s_im1])  # * here, why ?
    yemb_im1 = fs[-1](y_im1)
    hi = fs[1](*[yemb_im1, s_im1])  # * here, why ?
    e = time.time()
    thi = (e - s)

    s = time.time()
    pi, ai = fs[2](*[ctx, c_x, hi])
    e = time.time()
    tai = (e - s)

    s = time.time()
    si = fs[3](*[hi, ai])
    e = time.time()
    tsi = (e - s)

    s = time.time()
    mo = fs[4](*[yemb_im1, si, ai])
    e = time.time()
    tmo = (e - s)

    s = time.time()
    if args.use_mv:
        nvs = fs[5](*[mo, np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])])
    else:
        nvs = fs[5](mo)
    e = time.time()
    tvs = (e - s)

    s = time.time()
    ps = fs[6](*[mo, 10])
    e = time.time()
    tps = (e - s)

    s = time.time()
    p = fs[7](*[nvs])
    e = time.time()
    tp = (e - s)

    s = time.time()
    p = fs[8](*[y_im1, ctx, s_im1])
    e = time.time()
    tnext = (e - s)

    total = tinit + thi + tai + tsi + tmo + tvs + tp
    debug('{:16} {:7} {}%'.format('init', format(tinit, '0.4f'), (tinit / total) * 100))
    debug('{:16} {:7} {}%'.format('first hidden', format(thi, '0.4f'), (thi / total) * 100))
    debug('{:16} {:7} {}%'.format('attention', format(tai, '0.4f'), (tai / total) * 100))
    debug('{:16} {:7} {}%'.format('second hidden', format(tsi, '0.4f'), (tsi / total) * 100))
    debug('{:16} {:7} {}%'.format('merge out', format(tmo, '0.4f'), (tmo / total) * 100))
    debug('{:16} {:7} {}%'.format('next scores', format(tvs, '0.4f'), (tvs / total) * 100))
    debug('{:16} {:7} {}%'.format('softmax', format(tp, '0.4f'), (tp / total) * 100))

    debug('{:15} {:7}'.format('one slice scores', format(tps, '0.4f')))
    debug('{:15} {:7}'.format('f_next', format(tnext, '0.4f')))

    debug('\tload source and target vocabulary ...')
    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    debug('\tvocabulary contains <S>, <UNK> and </S>')

    seos_idx, teos_idx = config['src_vocab_size'] - 1, config['trg_vocab_size'] - 1
    src_vocab = ensure_special_tokens(
        src_vocab, bos_idx=0, eos_idx=seos_idx, unk_idx=config['unk_id'])
    trg_vocab = ensure_special_tokens(
        trg_vocab, bos_idx=0, eos_idx=teos_idx, unk_idx=config['unk_id'])

    # the trg_vocab is originally:
    #   {'UNK': 1, '<s>': 0, '</s>': 0, 'is': 5, ...}
    # after ensure_special_tokens, the trg_vocab becomes:
    #   {'<UNK>': 1, '<S>': 0, '</S>': trg_vocab_size-1, 'is': 5, ...}
    trg_vocab_i2w = {index: word for word, index in trg_vocab.iteritems()}
    src_vocab_i2w = {index: word for word, index in src_vocab.iteritems()}
    # after reversing, the trg_vocab_i2w become:
    #   {1: '<UNK>', 0: '<S>', trg_vocab_size-1: '</S>', 5: 'is', ...}
    debug('\t~done source vocab count: {}, target vocab count: {}'.format(
        len(src_vocab), len(trg_vocab)))

    lm = kenlm.Model(lmpath) if (lmpath and search_mode == 3) else None

    val_prefix = config['val_prefix']
    config['val_prefix'] = valid_set
    config['val_set'] = config['val_tst_dir'] + config['val_prefix'] + '.src'
    if config['val_prefix'] == val_prefix:
        dev_stream = get_dev_stream(**config)
    else:
        dev_stream = get_tst_stream(**config)
    if args.use_valid:
        _log('start decoding ... {}'.format(config['val_set']))
    trans.load(model_name)  # this is change all weights of nmt, importance!!!
    np_params = trans.load2numpy(model_name)
    '''
    for np_param in np_params.files:
        print type(np_param)
        print np_param
    '''
    params = trans.params
    # _log('Weights in model {}'.format(model_name))
    # for shared_var in params:
    #    _log('{} : {} {} {}'.format(shared_var.name, shared_var.get_value().sum(),
    #                                       type(shared_var), type(shared_var.get_value())))

    translator = Translator(
        fs=fs,
        switchs=switchs,
        mode=search_mode,
        svcb=src_vocab,
        svcb_i2w=src_vocab_i2w,
        tvcb=trg_vocab,
        tvcb_i2w=trg_vocab_i2w,
        ngram=ngram,
        k=beam_size,
        thresh=kl,
        lm=lm,
        ln_alpha=alpha,
        cp_beta=beta
    )

    if not args.use_valid:
        # s = np.asarray([[0, 10811, 140, 217, 19, 1047, 482, 29999, 0, 0, 0]])
        # 章启月 昨天 也 证实 了 俄罗斯 媒体 的 报道 , 说 中国 国家 主席 江泽民 前晚 应 俄罗斯 总统
        # 普京 的 要求 与 他 通 了 电话 , 双方 主要 是 就中 俄 互利 合作 问题 交换 了 意见 。
        '''
        s = np.asarray([[3490, 1477, 41, 1711, 10, 422, 722, 3, 433, 2, 28, 11, 39, 161, 240, 1,
                         219, 422, 217, 1512, 3, 120, 19, 32, 3958, 10, 630, 2, 158, 147, 8,
                         11963, 651, 1185, 51, 36, 882, 10, 267, 4, 29999]])
        '''
        s = np.asarray([[334, 1212, 2, 126, 3, 1, 27, 1, 11841, 2358, 5313, 2621, 10312, 2564,
                         100, 316, 21219, 2, 289, 18, 680, 11, 3161, 3, 316, 21219, 2, 41, 18,
                         365, 680, 316, 7, 772, 3, 60, 2, 147, 1275, 316, 1, 6737, 17, 11608, 50,
                         5284, 2, 279, 84, 8635, 1, 2, 569, 3246, 680, 388, 342, 2, 84, 285,
                         4897, 41, 4144, 11996, 4, 29999]])
        # 今天 天气 不错 </S>
        # s = np.asarray([[153, 1660, 5137, 29999]])
        # s = np.asarray([[3490]])
        t = np.asarray([[0, 10782, 2102, 1735, 4, 1829, 1657, 29999, 0]])
        pv = np.asarray([0, 10782, 2102, 1735, 4, 1829, 1657, 29999])
        translator.trans_samples(s, t)
        sys.exit(0)

    # trans sentece
    viter = dev_stream.get_epoch_iterator()

    avg_merges_rate, trans = translator.single_trans_valid(viter)
    # trans = translator.multi_process(viter, n_process=nprocess)

    outdir = args.workspace

    init_dir(outdir)

    outprefix = outdir + '/trans'
    valid_out = "{}_e{}_upd{}_b{}m{}_kl{}bch{}_ln{}_cp{}".format(
        outprefix, epoch, batch, beam_size, search_mode, kl, args.use_batch, alpha, beta)
    fVal_save = open(valid_out, 'w')    # valids/trans
    fVal_save.writelines(trans)
    fVal_save.close()

    mteval_bleu, multi_bleu = valid_bleu(valid_out, config['val_tst_dir'], config['val_prefix'])
    mteval_bleu = float(mteval_bleu)

    score_file_name = '{}/mteval_bleu.pkl'.format(outdir)
    scores = []
    if os.path.exists(score_file_name):
        with open(score_file_name) as score_file:
            scores = pickle.load(score_file)
            score_file.close()
    scores.append(mteval_bleu)
    with open(score_file_name, 'w') as score_file:
        pickle.dump(scores, score_file)
        score_file.close()
    if mteval_bleu == max(scores) and args.when_train == 1:  # current bleu is maximal in history
        child = subprocess.Popen(
            'cp {} {}/params.best.npz'.format(model_name, outdir), shell=True)
        with open('{}/b{}m{}kl{}.log'.format(outdir, beam_size, search_mode, kl), 'a') as logfile:
            logfile.write('epoch [{}], batch[{}], BLEU score : {}'.format(
                epoch, batch, mteval_bleu))
            logfile.write('\n')
            logfile.close()

    # ori_mteval_bleu, ori_multi_bleu = fetch_bleu_from_file(oriref_bleu_log)
    sfig = '{}.{}'.format(outprefix, 'sfig')
    # sfig_content = str(eidx) + ' ' + str(uidx) + ' ' + str(mteval_bleu) + ' ' + \
    #    str(multi_bleu) + ' ' + str(ori_mteval_bleu) + ' ' + str(ori_multi_bleu)
    sfig_content = ('{} {} {} {} {} {} {} {} {} {}').format(
        alpha,
        beta,
        epoch,
        batch,
        search_mode,
        beam_size,
        kl,
        mteval_bleu,
        multi_bleu,
        avg_merges_rate
    )
    append_file(sfig, sfig_content)

    os.rename(valid_out, "{}_{}_{}.txt".format(valid_out, mteval_bleu, multi_bleu))
