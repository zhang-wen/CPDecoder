from __future__ import division

import time
import sys
import numpy

from utils import _filter_reidx, part_sort, exeTime, kl_dist, \
    euclidean, init_beam, back_tracking, log, debug
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

        self.beam = []
        self.translations = []

    def cube_prune_trans(self, src_sent):

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

        log('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            src_sent_len, len(best_trans), self.maxlen, best_loss))
        log('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] ce[{}]'.format(*self.lqc[0:8]))

        avg_merges = format(self.lqc[9] / self.lqc[8], '0.3f')
        log('average merge count[{}/{}={}]'.format(self.lqc[9],
                                                   self.lqc[8], avg_merges))

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
            score_im1_r0, s_im1_r0, y_im1, y_im2, y_im3, _ = leq_class[0]
            subcube = []
            subcube_line_mergeout = []
            _avg_si, _avg_hi, _avg_ai, _avg_scores_i = None, None, None, None
            _cube_next_lm_krank_ces, _cube_next_krank_ces = None, None

            if each_subcube_rowsz == 1:
                _avg_sim1 = s_im1_r0
            else:
                merged_sim1 = [tup[1] for tup in leq_class]
                _avg_sim1 = numpy.mean(numpy.array(merged_sim1), axis=0)
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

                lm_next_logps, next_wids = vocab_prob_given_ngram(
                    self.lm, gram, self.tvcb, self.tvcb_i2w)
                np_lm_next_logps = numpy.asarray(lm_next_logps)
                np_next_wids = numpy.asarray(next_wids)

                np_lm_next_neg_logps = -np_lm_next_logps
                next_krank_ids = part_sort(np_lm_next_neg_logps, self.k - cnt_transed)
                _cube_next_lm_krank_ces = np_lm_next_neg_logps[next_krank_ids]
                next_krank_wids = np_next_wids[next_krank_ids]

                for idx in gram:
                    log(idx if idx == -1 else self.tvcb_i2w[idx] + ' ', nl=False)
                log('=> ', nl=False)
                for wid in next_krank_wids:
                    log('{}({}) '.format(self.tvcb_i2w[wid], np_lm_next_neg_logps[wid]), nl=False)
                log('')
                self.approx_items.append(None)
            else:
                # TODO sort the row dimension by average scores
                debug('sort by averge scores')
                y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                _, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)
                _avg_scores_i = self.fn_pws(_avg_moi, self.ptv)  # the larger the better

                if self.ifscore:
                    _next_ces_flat = -_avg_scores_i.flatten()    # (1,vocsize) -> (vocsize,)
                else:
                    _next_ces = self.fn_ce(_avg_scores_i)
                    _next_ces_flat = _next_ces.flatten()    # (1,vocsize) -> (vocsize,)

                next_krank_ids = part_sort(_next_ces_flat, self.k - cnt_transed)
                next_krank_wids = next_krank_ids
                _cube_next_krank_ces = _next_ces_flat[next_krank_wids]

                log(str(y_im1) + '' if y_im1 == -1 else self.tvcb_i2w[y_im1] + ' ', nl=False)
                log('=> ', nl=False)
                for wid in next_krank_wids:
                    log('{}({}) '.format(self.tvcb_i2w[wid], _next_ces_flat[wid]), nl=False)
                log('')
                self.approx_items.append((y_emb_im1, _avg_hi, _avg_ai, _avg_si))

            # add cnt for error The truth value of an array with more than one element is ambiguous
            for i, tup in enumerate(leq_class):
                subcube.append([tup +
                                (_avg_sim1,
                                 None if _cube_next_lm_krank_ces is None else _cube_next_lm_krank_ces[
                                     j],
                                 None if _cube_next_krank_ces is None else _cube_next_krank_ces[j],
                                 wid, i, j, whichsubcub, each_subcube_rowsz) for j, wid in
                                enumerate(next_krank_wids)])
                subcube_line_mergeout.append(None)

            cube.append(subcube)
            self.cube_lines_mergeout.append(subcube_line_mergeout)

        # print created cube before generating current beam for debug ...
        debug('\n************************************************')
        nsubcube = len(cube)
        for subcube_id in xrange(nsubcube):
            subcube = cube[subcube_id]
            nmergings = len(subcube)
            debug('group: {} contains {} mergings:'.format(subcube_id, nmergings))
            for mergeid in xrange(nmergings):
                line_in_subcube = subcube[mergeid]
                score_im1 = line_in_subcube[0][0]
                debug('{: >7} => '.format(format(score_im1, '0.3f')), nl=False)
                for cubetup in line_in_subcube:
                    lm_score = cubetup[-7]
                    model_score = cubetup[-6]
                    debug('{: >5}|{: >5}, '.format(
                        None if lm_score is None else format(lm_score, '0.3f'),
                        None if model_score is None else format(model_score, '0.3f')), nl=False)
                debug(' => ', nl=False)
                for cubetup in line_in_subcube:
                    wid = cubetup[-5]
                    debug('{}|{} @@ '.format(wid, self.tvcb_i2w[wid]), nl=False)
                debug('')
        debug('************************************************\n')

        return cube

    ##################################################################

    # NOTE: (Wen Zhang) Given cube, we calculate true score,
    # computation-expensive here

    ##################################################################

    def Push_heap(self, heap, bidx, citem):
        score_im1, s_im1, y_im1, y_im2, y_im3, bp, _avg_sim1, \
            _avg_lm_ces_i, _avg_ces_i, yi, iexp, jexp, which, rsz = citem

        if self.lm is not None and bidx >= 4:
            debug('self.lm is not None and bidx >= 4')
            if self.ifapprox_dist:
                debug('approximated whole distribution ...')

                if self.approx_items[which] is not None:
                    (_avg_si, _avg_moi) = self.approx_items[which]
                else:
                    y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                    _avg_pi, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                    _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                    _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)
                    self.approx_items[which] = (_avg_si, _avg_moi)

                _avg_ces_ith = -self.fn_one(_avg_moi, yi).flatten()[0]
                true_si = _avg_si

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
                if self.ifavg_att:
                    debug('approximated attention distribution ...')

                    if self.cube_lines_mergeout[which][iexp] is not None:
                        true_si, _avg_moi = self.cube_lines_mergeout[which][iexp]
                    else:
                        if self.approx_items[which] is not None:
                            (y_emb_im1, _avg_hi, _avg_ai) = self.approx_items[which]
                        else:
                            y_emb_im1, _avg_hi = self.fn_nh(y_im1, _avg_sim1)
                            _avg_pi, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                            self.approx_items[which] = (y_emb_im1, _avg_hi, _avg_ai)

                        true_si = _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                        _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)
                        self.cube_lines_mergeout[which][iexp] = true_si, _avg_moi

                    if self.ifscore:
                        _avg_ces_ith = -self.fn_one(_avg_moi, yi).flatten()[0]
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
                        sci = self.fn_pws(_avg_moi, self.ptv)
                        cei = self.fn_ce(sci)
                        if self.ifadd_lmscore:
                            true_sci = score_im1 + cei.flatten()[yi] + _avg_lm_ces_i
                            debug('S| {}={}+{}+{}'.format(format(true_sci, '0.3f'),
                                                          format(score_im1, '0.3f'),
                                                          format(cei.flatten()[yi], '0.3f'),
                                                          format(_avg_lm_ces_i, '0.3f')))
                        else:
                            true_sci = score_im1 + cei.flatten()[yi]
                            debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                       format(score_im1, '0.3f'),
                                                       format(cei.flatten()[yi], '0.3f')))

                else:
                    debug('real distribution ...')

                    if self.cube_lines_mergeout[which][iexp] is not None:
                        true_si, moi = self.cube_lines_mergeout[which][iexp]
                    else:
                        y_emb_im1, hi = self.fn_nh(y_im1, s_im1)
                        ai = self.fn_na(self.context, self.uh, hi)[1]
                        true_si = self.fn_ns(hi, ai)
                        moi = self.fn_mo(y_emb_im1, ai, true_si)
                        self.cube_lines_mergeout[which][iexp] = true_si, moi

                    if self.ifscore:
                        true_ces_ith = -self.fn_one(moi, yi).flatten()[0]
                        if self.ifadd_lmscore:
                            true_sci = score_im1 + true_ces_ith + _avg_lm_ces_i
                            debug('S| {}={}+{}+{}'.format(format(true_sci, '0.3f'),
                                                          format(score_im1, '0.3f'),
                                                          format(true_ces_ith, '0.3f'),
                                                          format(_avg_lm_ces_i, '0.3f')))
                        else:
                            true_sci = score_im1 + true_ces_ith
                            debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                       format(score_im1, '0.3f'),
                                                       format(true_ces_ith, '0.3f')))
                    else:
                        sci = self.fn_pws(moi, self.ptv)
                        cei = self.fn_ce(sci)
                        if self.ifadd_lmscore:
                            true_sci = score_im1 + cei.flatten()[yi] + _avg_lm_ces_i
                            debug('S| {}={}+{}+{}'.format(format(true_sci, '0.3f'),
                                                          format(score_im1, '0.3f'),
                                                          format(cei.flatten()[yi], '0.3f'),
                                                          format(_avg_lm_ces_i, '0.3f')))
                        else:
                            true_sci = score_im1 + cei.flatten()[yi]
                            debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                       format(score_im1, '0.3f'),
                                                       format(cei.flatten()[yi], '0.3f')))
        else:
            debug('self.lm is None or bidx < 4')
            assert(self.approx_items[which] is not None)
            (y_emb_im1, _avg_hi, _avg_ai, _avg_si) = self.approx_items[which]
            if rsz == 1 or self.ifapprox_dist:
                debug('row size == 1 or approximated whole distribution ...')
                true_si = _avg_si
                true_sci = score_im1 + _avg_ces_i
                debug('A| {}={}+{}'.format(format(true_sci, '0.3f'),
                                           format(score_im1, '0.3f'),
                                           format(_avg_ces_i, '0.3f')))
            else:
                if self.ifavg_att:
                    debug('approximated attention distribution ...')

                    if self.cube_lines_mergeout[which][iexp] is not None:
                        true_si, _avg_moi = self.cube_lines_mergeout[which][iexp]
                    else:
                        if self.approx_items[which] is not None:
                            (y_emb_im1, _avg_hi, _avg_ai) = self.approx_items[which]
                        else:
                            self.approx_items[which] = (y_emb_im1, _avg_hi, _avg_ai)

                        true_si = _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                        _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)
                        self.cube_lines_mergeout[which][i] = _avg_moi, true_si

                    if self.ifscore:
                        _avg_ces_ith = -self.fn_one(_avg_moi, yi).flatten()[0]
                        true_sci = score_im1 + _avg_ces_ith
                        debug('S| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                   format(score_im1, '0.3f'),
                                                   format(_avg_ces_ith, '0.3f')))
                    else:
                        sci = self.fn_pws(_avg_moi, self.ptv)
                        cei = self.fn_ce(sci)
                        true_sci = score_im1 + cei.flatten()[yi]
                        debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                   format(score_im1, '0.3f'),
                                                   format(cei.flatten()[yi], '0.3f')))
                else:
                    debug('real distribution ...')

                    if self.cube_lines_mergeout[which][iexp] is not None:
                        true_si, moi = self.cube_lines_mergeout[which][iexp]
                    else:
                        y_emb_im1, hi = self.fn_nh(y_im1, s_im1)
                        ai = self.fn_na(self.context, self.uh, hi)[1]
                        true_si = self.fn_ns(hi, ai)
                        moi = self.fn_mo(y_emb_im1, ai, true_si)
                        self.cube_lines_mergeout[which][iexp] = true_si, moi

                    if self.ifscore:
                        true_ces_ith = -self.fn_one(moi, yi).flatten()[0]
                        true_sci = score_im1 + true_ces_ith
                        debug('S| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                   format(score_im1, '0.3f'),
                                                   format(true_ces_ith, '0.3f')))
                    else:
                        sci = self.fn_pws(moi, self.ptv)
                        cei = self.fn_ce(sci)

                        true_sci = score_im1 + cei.flatten()[yi]
                        debug('P| {}={}+{}'.format(format(true_sci, '0.3f'),
                                                   format(score_im1, '0.3f'),
                                                   format(cei.flatten()[yi], '0.3f')))

        heapq.heappush(heap, (true_sci, next(self.cnt), true_si, yi, bp, iexp, jexp, which))

    ##################################################################

    # NOTE: (Wen Zhang) cube pruning

    ##################################################################

    def cube_prune(self, bidx, cube):
        # search in cube
        # cube (matrix(mergings) or vector(no mergings))
        nsubcube = len(cube)
        each_subcube_colsz, each_subcube_rowsz = [], []
        cube_size, counter = 0, 0
        extheap, wavetag, buf_state_merge = [], [], []
        self.lqc[8] += nsubcube   # count of total sub-cubes
        for whichsubcube in xrange(nsubcube):
            subcube = cube[whichsubcube]
            rowsz = len(subcube)
            each_subcube_rowsz.append(rowsz)
            each_subcube_colsz.append(len(subcube[0]))
            # print bidx, rowsz
            self.lqc[9] += rowsz   # count of total lines in sub-cubes
            # initial heap, starting from the left-top corner (best word) of each subcube
            # real score here ... may adding language model here ...
            # we should calculate the real score in current beam when pushing
            # into heap
            self.Push_heap(extheap, bidx, subcube[0][0])
            buf_state_merge.append([])

        cnt_transed = len(self.translations)
        while len(extheap) > 0 and counter < self.k - cnt_transed:
            true_sci, _, true_si, yi, bp, iexp, jexp, which = heapq.heappop(extheap)
            if yi == self.eos_id:
                # beam items count decrease 1
                if self.ifnorm:
                    self.translations.append(
                        ((true_sci / bidx), true_sci, yi, bp, bidx))
                else:
                    self.translations.append(true_sci, yi, bp, bidx)
                debug('add sample {}'.format(self.translations[-1]))
                if len(self.translations) == self.k:
                    # last beam created and finish cube pruning
                    return True
            else:
                # generate one item in current beam
                self.locrt[0] += (bp + 1)
                self.locrt[1] += 1
                self.beam[bidx].append((true_sci, true_si, yi, bp))

            whichsubcub = cube[which]
            # make sure we do not add repeadedly
            if jexp + 1 < each_subcube_colsz[which]:
                right = whichsubcub[iexp][jexp + 1]
                self.Push_heap(extheap, bidx, right)
            if iexp + 1 < each_subcube_rowsz[which]:
                down = whichsubcub[iexp + 1][jexp]
                self.Push_heap(extheap, bidx, down)
            counter += 1
        return False

    @exeTime
    def cube_pruning(self):

        for bidx in range(1, self.maxlen + 1):

            eq_classes = OrderedDict()
            self.approx_items, self.cube_lines_mergeout = [], []
            self.merge(bidx, eq_classes)

            # create cube and generate next beam from cube
            cube = self.create_cube(bidx, eq_classes)

            if self.cube_prune(bidx, cube):
                log('early stop! see {} samples ending with EOS.'.format(self.k))
                avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                log('average location of back pointers [{}/{}={}]'.format(
                    self.locrt[0], self.locrt[1], avg_bp))
                sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                best_sample = sorted_samples[0]
                log('translation length(with EOS) [{}]'.format(best_sample[-1]))
                for sample in sorted_samples:  # tuples
                    log('{}'.format(sample))

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
        log('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            log('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(maxlen))
            best_sample = (self.beam[maxlen][0][0],) + \
                self.beam[maxlen][0][2:] + (maxlen, )
            log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample, False)
        else:
            log('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                log('{}'.format(sample))
            return back_tracking(self.beam, best_sample, False)
