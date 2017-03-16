from __future__ import division

import time
import sys
import numpy

from utils import _filter_reidx, part_sort, exeTime, kl_dist, \
    euclidean, init_beam, back_tracking, _log, debug
from collections import OrderedDict
import heapq
from itertools import count
import copy
from search_bs import Func


from wlm import vocab_prob_given_ngram


class WCP(Func):

    def __init__(self, fs, switchs, bos_id=0, ngram=3, tvcb=None, tvcb_i2w=None, k=10, thresh=100.0,
                 lm=None, ptv=None):

        self.lqc = [0] * 10
        super(WCP, self).__init__(self.lqc, fs, switchs)

        self.cnt = count()

        self.ngram = ngram
        self.tvcb = tvcb
        self.tvcb_i2w = tvcb_i2w

        self.bos_id = bos_id
        self.eos_id = len(self.tvcb) - 1

        self.k = k
        self.locrt = [0] * 2
        self.thresh = thresh
        self.lm = lm
        self.ptv = ptv
        self.onerow_subcube_cnt = 0
        self.push_cnt = 0
        self.pop_cnt = 0
        self.down_pop_cnt = 0
        self.right_pop_cnt = 0

        self.beam = []

    def cube_prune_trans(self, src_sent):

        self.translations = []
        src_sent = src_sent[0] if self.ifvalid else src_sent  # numpy ndarray

        self.ptv = numpy.asarray(src_sent[1], dtype='int32') if self.ifvalid else None
        np_src_sent = numpy.asarray(src_sent, dtype='int64')
        if np_src_sent.ndim == 1:  # x (5,)
            # x(5, 1), (src_sent_len, batch_size)
            np_src_sent = np_src_sent[:, None]

        src_sent_len = np_src_sent.shape[0]
        self.maxlen = 2 * src_sent_len     # x(src_sent_len, batch_size)

        s_im1, self.context, self.uh = self.fn_init(np_src_sent)   # np_src_sent (sl, 1), beam==1

        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=self.maxlen, init_state=s_im1, detail=False)

        best_trans, best_loss = self.cube_pruning()

        _log('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            src_sent_len, len(best_trans), self.maxlen, best_loss))
        _log('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] ce[{}]'.format(*self.lqc[0:8]))

        avg_merges = format(self.lqc[9] / self.lqc[8], '0.3f')
        _log('average merge count[{}/{}={}]'.format(self.lqc[9],
                                                   self.lqc[8], avg_merges))
        _log('push count {}'.format(self.push_cnt))
        _log('pop count {}'.format(self.pop_cnt))
        _log('down pop count {}'.format(self.down_pop_cnt))
        _log('right pop count {}'.format(self.right_pop_cnt))
        _log('one row subcube count {}'.format(self.onerow_subcube_cnt))

        return _filter_reidx(self.bos_id, self.eos_id, best_trans, self.tvcb_i2w,
                             self.ifmv, self.ptv)

    ##################################################################

    # NOTE: merge all candidates in previous beam by euclidean distance of two state vector or KL
    # distance of alignment probability

    ##################################################################

    def merge(self, bidx, eq_classes):

        prevb = self.beam[bidx - 1]
        len_prevb = len(prevb)
        used = []
        key = 0

        _memory = [None] * len_prevb
        _mem_p = [None] * len_prevb
        for j in range(len_prevb):  # index of each item in last beam
            if j in used:
                continue

            tmp = []
            if _memory[j]:
                _needed = _memory[j]
                score_im1_1, s_im1_1, y_im1_1, y_im2_1, y_im3_1, nj = _needed
                if self.ifwatch_adist and _mem_p[j]:
                    pi_1 = _mem_p[j]
                assert(j == nj)
            else:
                # calculation
                score_im1_1, s_im1_1, y_im1_1, bp_im1_1 = prevb[j]
                (y_im2_1, bp_im2_1) = (-1, -
                                       1) if bidx < 2 else self.beam[bidx - 2][bp_im1_1][-2:]
                y_im3_1 = -1 if bidx < 3 else self.beam[bidx - 3][bp_im2_1][-2]

                if self.ifwatch_adist:
                    y_emb_im1_1, hi_1 = self.fn_nh(y_im1_1, s_im1_1)
                    pi_1, ai_1 = self.fn_na(ctx0, self.uh, hi_1)
                    _mem_p[j] = pi_1

                _needed = _memory[j] = (
                    score_im1_1, s_im1_1, y_im1_1, y_im2_1, y_im3_1, j)

            tmp.append(_needed)

            for jj in range(j + 1, len_prevb):

                if _memory[jj]:
                    _needed = _memory[jj]
                    score_im1_2, s_im1_2, y_im1_2, y_im2_2, y_im3_2, njj = _needed
                    if self.ifwatch_adist and _mem_p[jj]:
                        pi_2 = _mem_p[jj]
                    assert(jj == njj)
                else:   # calculation
                    score_im1_2, s_im1_2, y_im1_2, bp_im1_2 = prevb[jj]
                    (y_im2_2, bp_im2_2) = (-1, -1) if bidx < 2 else \
                        self.beam[bidx - 2][bp_im1_2][-2:]
                    y_im3_2 = -1 if bidx < 3 else self.beam[bidx - 3][bp_im2_2][-2]

                    if self.ifwatch_adist:
                        y_emb_im1_2, hi_2 = self.fn_nh(y_im1_2, s_im1_2)
                        pi_2, ai_2 = self.fn_na(ctx0, self.uh, hi_2)
                        _mem_p[jj] = pi_2

                    _needed = _memory[jj] = (
                        score_im1_2, s_im1_2, y_im1_2, y_im2_2, y_im3_2, jj)

                if self.merge_way == 'Him1':

                    distance = euclidean(s_im1_2, s_im1_1)

                    if self.ngram == 2:
                        debug('y11 y12 {} {}, {} {}'.format(y_im1_1, y_im1_2, distance,
                                                            self.thresh))
                        ifmerge = ((y_im1_2 == y_im1_1)
                                   and (distance < self.thresh))
                    elif self.ngram == 3:
                        debug('y21 y22 {} {}, y11 y12 {} {}, {} {}'.format(
                            y_im2_1, y_im2_2, y_im1_1, y_im1_2, distance, self.thresh))
                        ifmerge = ((y_im2_2 == y_im2_1) and (
                            y_im1_2 == y_im1_1) and (distance < self.thresh))
                    elif self.ngram == 4:
                        debug('y31 y32 {} {}, y21 y22 {} {}, y11 y12 {} {}, {} {}'.format(
                            y_im3_1, y_im3_2, y_im2_1, y_im2_2, y_im1_1, y_im1_2, distance,
                            self.thresh))
                        ifmerge = ((y_im3_2 == y_im3_1) and (y_im2_2 == y_im2_1)
                                   and (y_im1_2 == y_im1_1) and (distance < self.thresh))
                    else:
                        raise NotImplementedError

                elif self.merge_way == 'Hi':
                    raise NotImplementedError
                    ifmerge = (y_im1_2 == y_im1_1 and euclidean(
                        hi_2, hi_1) < self.thresh)
                elif self.merge_way == 'AiKL':
                    raise NotImplementedError
                    dist = kl_dist(pi_2, pi_1)
                    debug('attention prob kl distance: {}'.format(dist))
                    ifmerge = (y_im1_2 == y_im1_1 and dist < self.thresh)

                if ifmerge:
                    tmp.append(_needed)
                    used.append(jj)

                if self.ifwatch_adist:
                    dist = kl_dist(pi_2, pi_1)
                    debug('{} {} {}'.format(j, jj, dist))

            eq_classes[key] = tmp
            key += 1

    ##################################################################

    # NOTE: (Wen Zhang) create cube by sort row dimension

    ##################################################################

    #@exeTime
    def create_cube(self, bidx, eq_classes):
        # eq_classes: (score_im1, y_im1, hi, ai, loc_in_prevb) NEW

        cube = []
        cnt_transed = len(self.translations)
        for whichsubcub, leq_class in eq_classes.iteritems():   # sub cube

            each_subcube_rowsz = len(leq_class)
            self.prev_beam_ptrs += each_subcube_rowsz
            #print self.prev_beam_ptrs
            #if bidx >= 2 and self.prev_beam_ptrs > self.avg_bp_by_cur_step + 5:
            #    return cube

            score_im1_r0, s_im1_r0, y_im1, y_im2, y_im3, _ = leq_class[0]
            subcube = []
            subcube_line_cache = []
            _avg_si, _avg_hi, _avg_ai, _avg_scores_i = None, None, None, None
            _cube_lm_krank_ces_i, _cube_krank_scores_i = None, None

            if each_subcube_rowsz == 1:
                _avg_sim1 = s_im1_r0
                self.onerow_subcube_cnt += 1
            else:
                merged_score_im1 = [tup[0] for tup in leq_class]
                merged_sim1 = [tup[1] for tup in leq_class[0:1]]
                np_merged_score_im1 = numpy.array(merged_score_im1, dtype='float32')
                np_merged_sim1 = numpy.array(merged_sim1)
                # arithmetic mean
                _avg_sim1 = numpy.mean(np_merged_sim1, axis=0)

                # geometric mean , not work
                #_avg_sim1 = numpy.power(numpy.prod(np_merged_sim1, axis=0), 1.0 /
                #                        np_merged_sim1.shape[0])

                # harmonic mean
                #_avg_sim1 = np_merged_sim1.shape[0] / numpy.sum(1.0 / np_merged_sim1, axis=0)

                # weighted harmonic mean
                #assert(np_merged_sim1.shape[0] == np_merged_score_im1.shape[0])
                #_avg_sim1 = numpy.sum(np_merged_score_im1, axis=0) / numpy.sum(
                #    np_merged_score_im1[:,None,None] / np_merged_sim1, axis=0)

                # weighted mean
                #exp_score_im1 = numpy.exp(np_merged_score_im1 -
                #                                    numpy.max(np_merged_score_im1, axis=0))
                #softmax_score_im1 = exp_score_im1 / exp_score_im1.sum()
                #_avg_sim1 = numpy.sum(softmax_score_im1[:,None,None] * np_merged_sim1, axis=0)

                # quadratic mean, not work
                #_avg_sim1 = numpy.power(numpy.mean(numpy.power(np_merged_sim1, 2), axis=0), 
                #                       1.0 / np_merged_sim1.shape[0])

                # 
                # for tup in leq_class: watch the attention prob pi dist here ....

            if self.lm is not None and bidx >= 4:
                # TODO sort the row dimension by language model words distribution
                debug('sort by lm: -3 -2 -1 => {} {} {}'.format(y_im3, y_im2, y_im1))
                if self.ngram == 2:
                    gram = [y_im1]
                elif self.ngram == 3:
                    gram = [y_im1] if y_im2 == -1 else [y_im2, y_im1]
                elif self.ngram == 4:
                    gram = [y_im1] if y_im3 == -1 and y_im2 == -1 else (
                        [y_im2, y_im1] if y_im3 == -1 else [y_im3, y_im2, y_im1])
                else:
                    raise NotImplementedError

                lm_next_logps, next_ids = vocab_prob_given_ngram(
                    self.lm, gram, self.tvcb, self.tvcb_i2w)
                np_lm_next_neg_logps = -numpy.asarray(lm_next_logps)
                np_next_ids = numpy.asarray(next_ids)

                _next_krank_ids = part_sort(np_lm_next_neg_logps, self.k - cnt_transed)
                _cube_lm_krank_ces_i = np_lm_next_neg_logps[_next_krank_ids]
                _next_krank_wids = np_next_ids[_next_krank_ids]

                for idx in gram:
                    _log(idx if idx == -1 else self.tvcb_i2w[idx] + ' ', nl=False)
                _log('=> ', nl=False)
                for wid in _next_krank_wids:
                    _log('{}({}) '.format(self.tvcb_i2w[wid], np_lm_next_neg_logps[wid]), nl=False)
                _log('')
                self.pop_subcube_approx_cache.append(None)
            else:
                # TODO sort the row dimension by average scores
                debug('sort by averge scores')
                _y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                _, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                _avg_moi = self.fn_mo(_y_emb_im1, _avg_ai, _avg_si)
                _avg_scores_i = self.fn_pws(_avg_moi, self.ptv)  # the larger the better
                _avg_ces_i = self.fn_ce(_avg_scores_i).flatten()
                _next_krank_wids = part_sort(_avg_ces_i, self.k - cnt_transed)
                _cube_krank_scores_i = _cube_krank_ces_ith = _avg_ces_i[_next_krank_wids]

                self.pop_subcube_approx_cache.append((_y_emb_im1, _avg_hi, _avg_ai, _avg_si,
                                                      _avg_moi, _avg_scores_i, _next_krank_wids,
                                                     _cube_krank_ces_ith))
            self.push_subcube_approx_cache.append(None)

            # add cnt for error The truth value of an array with more than one element is ambiguous
            for i, tup in enumerate(leq_class):
                subcube.append([tup +
                                (_avg_sim1,
                                 None if _cube_lm_krank_ces_i is None else _cube_lm_krank_ces_i[j],
                                 None if _cube_krank_scores_i is None else _cube_krank_scores_i[j],
                                 wid, i, j, whichsubcub, each_subcube_rowsz)
                                for j, wid in enumerate(_next_krank_wids)])
                subcube_line_cache.append(None)

            cube.append(subcube)
            self.subcube_lines_cache.append(subcube_line_cache)

        # print created cube before generating current beam for debug ...
        debug('\n################################ CUBE ################################')
        nsubcube = len(cube)
        debug('MERGE => ', nl=False)
        for subcube_id in xrange(nsubcube):
            nmergings = len(cube[subcube_id])
            debug('{} '.format(nmergings), nl=False)
        debug('')
        for subcube_id in xrange(nsubcube):
            subcube = cube[subcube_id]
            nmergings = len(subcube)
            debug('Group: {} contains {} mergings:'.format(subcube_id, nmergings))
            for mergeid in xrange(nmergings):
                line_in_subcube = subcube[mergeid]
                first_item = line_in_subcube[0]
                score_im1, y_im1 = first_item[0], first_item[2]
                y_im1_w = None if y_im1 == -1 else self.tvcb_i2w[y_im1]
                debug('{}={}({: >7}) => '.format(
                    y_im1, y_im1_w,
                    format(score_im1, '0.2f')), nl=False)
                for cubetup in line_in_subcube:
                    wid = cubetup[-5]
                    lm_score = cubetup[-7]
                    model_score = cubetup[-6]
                    debug('{}={}({: >5}&+{: >5}={: >5}) | '.format(
                        wid, self.tvcb_i2w[wid],
                        None if lm_score is None else format(lm_score, '0.2f'),
                        None if model_score is None else format(model_score, '0.2f'),
                        None if model_score is None else format(score_im1 + model_score, '0.2f')),
                        nl=False)
                debug('')
        debug('######################################################################')

        return cube

    ##################################################################

    # NOTE: (Wen Zhang) Given cube, we calculate true score,
    # computation-expensive here

    ##################################################################

    def Push_heap(self, heap, bidx, citem):

        score_im1, s_im1, y_im1, y_im2, y_im3, bp, _avg_sim1, \
            _avg_lm_ces_i, _cube_krank_scores_i, yi, iexp, jexp, which, rsz = citem

        if self.lm is not None and bidx >= 4:
            debug('self.lm is not None and bidx >= 4')
            if self.ifapprox_dist == 1:
                debug('totally approximated distribution ...')
                if self.push_subcube_approx_cache[which] is not None:
                    (_avg_si, _avg_ces_ith) = self.push_subcube_approx_cache[which]
                else:
                    _y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                    _avg_pi, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                    _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                    _avg_moi = self.fn_mo(_y_emb_im1, _avg_ai, _avg_si)
                    if self.ifscore:
                        _avg_ces_ith = -self.fn_one(_avg_moi, yi).flatten()[0]
                    else:
                        _avg_scores_ith = self.fn_pws(_avg_moi, self.ptv)
                        _avg_ces_ith = self.fn_ce(_avg_scores_ith).flatten()[yi]
                    self.push_subcube_approx_cache[which] = (_avg_si, _avg_ces_ith)
                true_si = _avg_si
            else:
                if self.ifapprox_att == 1:
                    debug('approximated attention distribution ...')
                    if self.subcube_lines_cache[which][iexp] is not None:
                        (_avg_si, _avg_ces_ith) = self.subcube_lines_cache[which][iexp]
                    else:
                        if self.push_subcube_approx_cache[which] is not None:
                            _avg_ai = self.push_subcube_approx_cache[which]
                        else:
                            _y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                            _avg_pi, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                            self.push_subcube_approx_cache[which] = _avg_ai

                        y_emb_im1, hi = self.fn_nh(y_im1, s_im1)
                        _avg_si = self.fn_ns(hi, _avg_ai)
                        _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)

                        if self.ifscore:
                            _avg_ces_ith = -self.fn_one(_avg_moi, yi).flatten()[0]
                        else:
                            _avg_scores_ith = self.fn_pws(_avg_moi, self.ptv)
                            _avg_ces_ith = self.fn_ce(_avg_scores_ith).flatten()[yi]
                        self.subcube_lines_cache[which][iexp] = (_avg_si, _avg_ces_ith)
                    true_si = _avg_si
                else:
                    debug('real distribution ...')
                    if self.subcube_lines_cache[which][iexp] is not None:
                        (true_si, _avg_ces_ith) = self.subcube_lines_cache[which][iexp]
                    else:
                        y_emb_im1, hi = self.fn_nh(y_im1, s_im1)
                        pi, ai = self.fn_na(self.context, self.uh, hi)
                        true_si = self.fn_ns(hi, ai)
                        moi = self.fn_mo(y_emb_im1, ai, true_si)
                        if self.ifscore:
                            _avg_ces_ith = true_ces_ith = -self.fn_one(moi, yi).flatten()[0]
                        else:
                            sci = self.fn_pws(moi, self.ptv)
                            _avg_ces_ith = true_ces_ith = self.fn_ce(sci).flatten()[yi]
                        self.subcube_lines_cache[which][iexp] = (true_si, _avg_ces_ith)

            if self.ifadd_lmscore:
                true_sci = score_im1 + _avg_ces_ith + _avg_lm_ces_i
                debug('S| {}={}+{}+{}'.format(format(true_sci, '0.3f'),
                                              format(score_im1, '0.3f'),
                                              format(_avg_ces_ith, '0.3f'),
                                              format(_avg_lm_ces_i, '0.3f')))
            else:
                true_sci = score_im1 + _avg_ces_ith
                debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(_avg_ces_ith, '0.3f')))
        else:
            #debug('self.lm is None or bidx < 4')
            assert(self.pop_subcube_approx_cache[which] is not None)
            (_y_emb_im1, _avg_hi, _avg_ai, _avg_si, _avg_moi, _avg_scores_i, \
             _next_krank_wids, _avg_ces_ith) = self.pop_subcube_approx_cache[which]
            _avg_ce_ith = _avg_ces_ith[jexp]
            _sci = score_im1 + _avg_ce_ith

            if self.ifscore:
                debug('Mean_S| {}={}+{}'.format(format(_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(_avg_ce_ith, '0.3f')))
            else:
                debug('Mean_P| {}={}+{}'.format(format(_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(_avg_ce_ith, '0.3f')))

        heapq.heappush(heap, (_sci, next(self.cnt), score_im1, y_im1, s_im1, _avg_ai, _avg_si, \
                              _avg_ce_ith, yi, bp, iexp, jexp, which, rsz))

    ##################################################################

    # NOTE: (Wen Zhang) cube pruning

    ##################################################################

    def cube_prune(self, bidx, cube):
        # search in cube (matrix(mergings) or vector(no mergings))
        nsubcube = len(cube)
        each_subcube_colsz, each_subcube_rowsz = [], []
        cube_size = 0
        extheap, ijs_push, prev_pop_ijs = [], [], []
        self.lqc[8] += nsubcube   # count of total sub-cubes
        debug('\n################################ PUSH HEAP ################################')
        #for whichsubcube in xrange(nsubcube if nsubcube < 5 else (self.k if self.k <= 5 else 5)):
        for whichsubcube in xrange(nsubcube):
            subcube = cube[whichsubcube]
            rowsz = len(subcube)
            each_subcube_rowsz.append(rowsz)
            each_subcube_colsz.append(len(subcube[0]))
            # print bidx, rowsz
            self.lqc[9] += rowsz   # count of total lines in sub-cubes
            # initial heap, starting from the left-top corner (best word) of each subcube
            # real score here ... may adding language model here ...
            # we should calculate the real score in current beam when pushing into heap
            self.Push_heap(extheap, bidx, subcube[0][0])
            self.push_cnt += 1
            ijs_push.append([])
            prev_pop_ijs.append((-1, 0))

        cnt_transed = len(self.translations)
        while len(extheap) > 0:
            debug('\n################################ POP HEAP ################################')
            _sci, _, score_im1, y_im1, s_im1, _avg_ai, _avg_si, _avg_ce_ith, yi, bp, iexp, jexp, \
                    which, rsz = heapq.heappop(extheap)
            # true_sci, _, true_si, yi, bp, iexp, jexp, which = heapq.heappop(extheap)
            assert(each_subcube_rowsz[which] == rsz)
            self.pop_cnt += 1

            if rsz == 1 or self.ifapprox_dist == 1:
                debug('row size == 1 or totally approximated distribution ...')
                true_si = _avg_si
                true_ces_ith = _avg_ce_ith
            else:
                if self.subcube_lines_cache[which][iexp] is not None:
                    (true_si, true_ces_ith) = self.subcube_lines_cache[which][iexp]
                else:
                    y_emb_im1, hi = self.fn_nh(y_im1, s_im1)
                    if self.ifapprox_att == 1:
                        debug('use approximated attention distribution ...')
                        ai = _avg_ai
                        true_si = _avg_si = self.fn_ns(hi, _avg_ai)
                    else:
                        debug('real distribution ...')
                        pi, ai = self.fn_na(self.context, self.uh, hi)
                        true_si = self.fn_ns(hi, ai)
                    moi = self.fn_mo(y_emb_im1, ai, true_si)
                if self.ifscore:
                    true_ces_ith = -self.fn_one(moi, yi).flatten()[0]
                else:
                    true_scores_ith = self.fn_pws(moi, self.ptv)
                    true_ces_ith = self.fn_ce(true_scores_ith).flatten()[yi]
                self.subcube_lines_cache[which][iexp] = (true_si, true_ces_ith)

            true_sci = score_im1 + true_ces_ith

            prev_pop_i, prev_pop_j = prev_pop_ijs[which]
            if iexp > prev_pop_i:
                debug('subcube {}: {}-{}-{}-{}, down shift pop ... {} {} {}'.format(
                    which, prev_pop_i, prev_pop_j, iexp, jexp, score_im1+_avg_ce_ith, true_sci, yi))
                assert(jexp <= prev_pop_j)
                self.down_pop_cnt += 1
            elif jexp > prev_pop_j:
                assert(iexp <= prev_pop_i)
                debug('subcube {}: {}-{}-{}-{}, right shift pop ... {} {} {}'.format(
                    which, prev_pop_i, prev_pop_j, iexp, jexp, score_im1+_avg_ce_ith, true_sci, yi))
                self.right_pop_cnt += 1
            prev_pop_ijs[which] = (iexp, jexp)

            if self.ifscore:
                debug('True_S| {}={}+{}'.format(format(true_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(true_ces_ith, '0.3f')))
            else:
                debug('True_P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(true_ces_ith, '0.3f')))

            if yi == self.eos_id:
                # beam items count decrease 1
                if self.ifnorm:
                    self.translations.append(((true_sci / bidx), true_sci, yi, bp, bidx))
                else:
                    self.translations.append((true_sci, true_sci, yi, bp, bidx))
                debug('add sample {}'.format(self.translations[-1]))
                if len(self.translations) == self.k:
                    # last beam created and finish cube pruning
                    return True
            else:
                # generate one item in current beam
                self.locrt[0] += (bp + 1)
                self.locrt[1] += 1
                self.avg_bp_by_cur_step = self.locrt[0] / self.locrt[1]
                self.beam[bidx].append((true_sci, true_si, yi, bp))

            enough = len(self.beam[bidx]) == (self.k - cnt_transed)
            if enough: return False

            debug('\n################################ PUSH HEAP ################################')
            whichsubcub = cube[which]
            # make sure we do not add repeadedly
            if jexp + 1 < each_subcube_colsz[which] and (iexp, jexp + 1) not in ijs_push[which]:
                right = whichsubcub[iexp][jexp + 1]
                ijs_push[which].append((iexp, jexp + 1))
                self.push_cnt += 1
                self.Push_heap(extheap, bidx, right)
            if iexp + 1 < each_subcube_rowsz[which] and (iexp + 1, jexp) not in ijs_push[which]:
                down = whichsubcub[iexp + 1][jexp]
                self.push_cnt += 1
                ijs_push[which].append((iexp + 1, jexp))
                self.Push_heap(extheap, bidx, down)
        return False

    @exeTime
    def cube_pruning(self):

        for bidx in range(1, self.maxlen + 1):

            eq_classes = OrderedDict()
            self.pop_subcube_approx_cache, self.push_subcube_approx_cache, \
                    self.subcube_lines_cache = [], [], []
            self.prev_beam_ptrs = 0
            self.avg_bp_by_cur_step = 1.

            self.merge(bidx, eq_classes)

            # create cube and generate next beam from cube
            cube = self.create_cube(bidx, eq_classes)

            if self.cube_prune(bidx, cube):
                _log('early stop! see {} samples ending with EOS.'.format(self.k))
                avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                _log('average location of back pointers [{}/{}={}]'.format(
                    self.locrt[0], self.locrt[1], avg_bp))
                sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                best_sample = sorted_samples[0]
                _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
                for sample in sorted_samples:  # tuples
                    _log('{}'.format(sample))

                return back_tracking(self.beam, best_sample, False)

            self.beam[bidx] = sorted(self.beam[bidx], key=lambda tup: tup[0])
            debug('beam {} ----------------------------'.format(bidx))
            for b in self.beam[bidx]:
                debug('{}'.format(b))
                # debug('{}'.format(b[0:1] + b[2:]))
            # because of the the estimation of P(f|abcd) as P(f|cd), so the generated beam by
            # cube pruning may out of order by loss, so we need to sort it again here
            # losss from low to high

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        _log('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            _log('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(self.maxlen))
            best_sample = (self.beam[self.maxlen][0][0],) + \
                self.beam[self.maxlen][0][2:] + (self.maxlen, )
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample, False)
        else:
            _log('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                _log('{}'.format(sample))
            return back_tracking(self.beam, best_sample, False)
