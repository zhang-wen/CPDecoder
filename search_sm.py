from __future__ import division

import time
import sys
import numpy

from utils import _filter_reidx, part_sort, exeTime, kl_dist, \
    euclidean, init_beam, init_beam_sm, back_tracking, _log, debug
from collections import OrderedDict
import heapq
from itertools import count
import copy
from search_bs import Func


from wlm import vocab_prob_given_ngram


class WCP(Func):

    def __init__(self, fs, switchs, bos_id=0, tvcb=None, tvcb_i2w=None, k=10, ptv=None):

        self.lqc = [0] * 11
        super(WCP, self).__init__(self.lqc, fs, switchs)

        self.cnt = count()

        self.tvcb = tvcb
        self.tvcb_i2w = tvcb_i2w

        self.bos_id = bos_id
        self.eos_id = len(self.tvcb) - 1

        self.k = k
        self.ptv = ptv

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

        y_emb_im1 = self.fn_emb([-1])
        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam_sm(self.beam, cnt=self.maxlen, init_state=s_im1, init_y_emb_im1=y_emb_im1)

        best_trans, best_loss = self.cube_pruning()

        avg_merges = self.lqc[10] / self.lqc[9]
        print self.lqc[10], self.lqc[9], avg_merges
        return avg_merges, _filter_reidx(self.bos_id, self.eos_id, best_trans, self.tvcb_i2w,
                             self.ifmv, self.ptv)

    ##################################################################

    # NOTE: merge all candidates in previous beam by euclidean distance of two state vector or KL
    # distance of alignment probability

    ##################################################################

    #@exeTime
    def merge(self, bidx, eq_classes):

        prevb = self.beam[bidx - 1]
        len_prevb = len(prevb)
        self.lqc[10] += len_prevb
        if len_prevb == 1:
            self.lqc[9] += len_prevb
            eq_classes[0] = [prevb[0]]
            return

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
                _, _, y_im1_1, y_emb_im1_1, nj = _memory[j]
                assert(j == nj)
            else:
                # calculation
                score_im1_1, s_im1_1, y_im1_1, y_emb_im1_1, bp_im1_1 = prevb[j]
                _needed = _memory[j] = (score_im1_1, s_im1_1, y_im1_1, y_emb_im1_1, j)

            tmp.append(_needed)

            for jj in range(j + 1, len_prevb):
                if jj in used:
                    continue
                if _memory[jj]:
                    _needed = _memory[jj]
                    _, _, y_im1_2, y_emb_im1_2, njj = _needed
                    assert(jj == njj)
                else:
                    score_im1_2, s_im1_2, y_im1_2, y_emb_im1_2, bp_im1_2 = prevb[jj]
                    _needed = _memory[jj] = (score_im1_2, s_im1_2, y_im1_2, y_emb_im1_2, jj)

                #print euclidean(y_emb_im1_1, y_emb_im1_2)
                #if y_im1_2 == y_im1_1:
                if euclidean(y_emb_im1_1, y_emb_im1_2) < 3:
                    print 'add, ', j, jj
                    tmp.append(_needed)
                    used.append(jj)

            eq_classes[key] = tmp
            key += 1
        self.lqc[9] += key # count of total sub-cubes

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
            score_im1_r0, s_im1_r0, y_im1, y_emb_im1, _ = leq_class[0]
            subcube = []
            subcube_line_cache = []
            _avg_si, _avg_hi, _avg_ai, _avg_scores_i = None, None, None, None
            _cube_krank_scores_i = None

            _avg_sim1 = s_im1_r0

            if self.ifsplit:
                _avg_hi = self.fn_nh(y_emb_im1, _avg_sim1)
                _, _avg_ai = self.fn_na(self.context, self.uh, _avg_hi)
                _avg_si = self.fn_ns(_avg_hi, _avg_ai)
                _avg_moi = self.fn_mo(y_emb_im1, _avg_ai, _avg_si)
                _avg_scores_i = self.fn_pws(_avg_moi, self.ptv)  # the larger the better
                _avg_probs_i = self.fn_ce(_avg_scores_i).flatten()
            else:
                _avg_probs_i, _avg_si = self.fn_next(*[y_im1, self.context, _avg_sim1])
                _avg_probs_i = _avg_probs_i.flatten()

            _next_krank_wids = part_sort(-_avg_probs_i, self.k - cnt_transed)
            _avg_ces_i = -numpy.log(_avg_probs_i[_next_krank_wids])
            _cube_krank_scores_i = _cube_krank_ces_ith = _avg_ces_i

            self.pop_subcube_approx_cache.append((_avg_ai, _avg_si, _cube_krank_ces_ith))
            self.push_subcube_approx_cache.append(None)

            # add cnt for error The truth value of an array with more than one element is ambiguous
            for i, tup in enumerate(leq_class):
                if i > 1: break
                subcube.append([tup +
                                (_avg_sim1,
                                 None if _cube_krank_scores_i is None else _cube_krank_scores_i[j],
                                 wid, i, j, whichsubcub, each_subcube_rowsz)
                                for j, wid in enumerate(_next_krank_wids)])
                subcube_line_cache.append(None)

            cube.append(subcube)
            self.subcube_lines_cache.append(subcube_line_cache)

        return cube

    ##################################################################

    # NOTE: (Wen Zhang) create cube in batch

    ##################################################################

    #@exeTime
    def create_cube_batch(self, bidx, eq_classes):
        # eq_classes: (score_im1, y_im1, hi, ai, loc_in_prevb) NEW
        cube = []
        cnt_transed = len(self.translations)
        batch_y_im1, batch_s_im1, batch_y_emb = [], [], []
        for whichsubcub, leq_class in eq_classes.iteritems():   # sub cube

            each_subcube_rowsz = len(leq_class)
            score_im1_r0, s_im1_r0, y_im1, y_emb_im1, _ = leq_class[0]
            if len(s_im1_r0.shape) == 2:
                s_im1_r0 = s_im1_r0[0]
            subcube_line_cache = []
            _cube_krank_scores_i = None

            batch_y_im1.append(y_im1)
            batch_s_im1.append(s_im1_r0)
            batch_y_emb.append(y_emb_im1[0])
            self.push_subcube_approx_cache.append(None)

        np_batch_s_im1 = numpy.array(batch_s_im1, dtype='float32')
        #np_batch_y_im1 = numpy.array(batch_y_im1)
        np_batch_y_emb = numpy.array(batch_y_emb, dtype='float32')
        subcube_num = len(batch_y_im1)
        ctx = numpy.tile(self.context, [subcube_num, 1])
        uh = numpy.tile(self.uh, [subcube_num, 1])
        if np_batch_s_im1.shape[0] == 1 and 3 == len(np_batch_s_im1.shape):
            np_batch_s_im1 = np_batch_s_im1[0]

        _avg_si, _avg_hi, _avg_ai, _avg_scores_i = None, None, None, None
        if self.ifsplit:
            _avg_hi = self.fn_nh(np_batch_y_emb, np_batch_s_im1)
            _, _avg_ai = self.fn_na(ctx, uh, _avg_hi)
            next_states = self.fn_ns(_avg_hi, _avg_ai)
            _avg_moi = self.fn_mo(np_batch_y_emb, _avg_ai, next_states)
            _avg_scores_i = self.fn_pws(_avg_moi, self.ptv)  # the larger the better
            next_probs = self.fn_ce(_avg_scores_i)
        else:
            next_probs, next_states = self.fn_next(*[batch_y_im1, ctx, np_batch_s_im1])

        for which in range(len(eq_classes)):
            _avg_sim1, leq_class, next_prob = batch_s_im1[which], \
                    eq_classes[which], next_probs[which]
            _avg_si = next_states if len(next_states) == 1 else next_states[which]
            each_subcube_rowsz = len(leq_class)
            next_prob_flat = next_prob.flatten()
            _next_krank_wids = part_sort(-next_prob_flat, self.k - len(self.translations))
            k_avg_loss_flat = -numpy.log(next_prob_flat[_next_krank_wids])

            self.pop_subcube_approx_cache.append((_avg_ai, _avg_si, k_avg_loss_flat))
            # add cnt for error The truth value of an array with more than one element is ambiguous
            subcube = []
            for i, tup in enumerate(leq_class):
                #if i > 1: break
                subcube.append([tup +
                                (_avg_sim1,
                                 k_avg_loss_flat[j],
                                 wid, i, j, which, each_subcube_rowsz)
                                for j, wid in enumerate(_next_krank_wids)])
                subcube_line_cache.append(None)

            cube.append(subcube)
            self.subcube_lines_cache.append(subcube_line_cache)

        self.printCube(cube)

        return cube

    def printCube(self, cube):
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
                    debug('{}={}(+{: >5}={: >5}) | '.format(
                        wid, self.tvcb_i2w[wid],
                        None if model_score is None else format(model_score, '0.2f'),
                        None if model_score is None else format(score_im1 + model_score, '0.2f')),
                        nl=False)
                debug('')
        debug('######################################################################')



    ##################################################################

    # NOTE: (Wen Zhang) Given cube, we calculate true score,
    # computation-expensive here

    ##################################################################

    def Push_heap(self, heap, bidx, citem):

        score_im1, s_im1, y_im1, yemb_im1, bp, _avg_sim1, \
            _cube_krank_scores_i, yi, iexp, jexp, which, rsz = citem
        assert(self.pop_subcube_approx_cache[which] is not None)
        _avg_ai, _avg_si, _avg_ces_ith = self.pop_subcube_approx_cache[which]
        _avg_ce_ith = _avg_ces_ith[jexp]
        _sci = score_im1 + _avg_ce_ith

        heapq.heappush(heap, (_sci, next(self.cnt), score_im1, y_im1, yemb_im1, s_im1, \
                              _avg_ai, _avg_si, _avg_ce_ith, yi, bp, iexp, jexp, which, rsz))

    ##################################################################

    # NOTE: (Wen Zhang) cube pruning

    ##################################################################

    def cube_prune(self, bidx, cube):
        # search in cube (matrix(mergings) or vector(no mergings))
        nsubcube = len(cube)
        each_subcube_colsz, each_subcube_rowsz = [], []
        extheap, ijs_push, prev_pop_ijs = [], [], []
        for whichsubcube in xrange(nsubcube):
            subcube = cube[whichsubcube]
            rowsz = len(subcube)
            each_subcube_rowsz.append(rowsz)
            each_subcube_colsz.append(len(subcube[0]))
            self.Push_heap(extheap, bidx, subcube[0][0])
            ijs_push.append([])
            prev_pop_ijs.append((-1, 0))

        cnt_transed = len(self.translations)
        while len(extheap) > 0:
            debug('\n################################ POP HEAP ################################')
            _sci, _, score_im1, y_im1, yemb_im1, s_im1, _avg_ai, _avg_si, _avg_ce_ith, \
                    yi, bp, iexp, jexp, which, rsz = heapq.heappop(extheap)
            #assert(each_subcube_rowsz[which] == rsz)

            if rsz == 1 or self.ifapprox_dist == 1:
                debug('row size == 1 or totally approximated distribution ...')
                true_si = _avg_si
                true_ces_ith = _avg_ce_ith
            else:
                if self.subcube_lines_cache[which][iexp] is not None:
                    (true_si, true_ces_ith) = self.subcube_lines_cache[which][iexp]
                else:
                    debug('real distribution ...')
                    if len(s_im1.shape) == 1:
                        s_im1 = s_im1.reshape((1, s_im1.shape[0]))
                    hi = self.fn_nh(yemb_im1, s_im1)
                    pi, ai = self.fn_na(self.context, self.uh, hi)
                    true_si = self.fn_ns(hi, ai)
                    moi = self.fn_mo(yemb_im1, ai, true_si)
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
            elif jexp > prev_pop_j:
                assert(iexp <= prev_pop_i)
                debug('subcube {}: {}-{}-{}-{}, right shift pop ... {} {} {}'.format(
                    which, prev_pop_i, prev_pop_j, iexp, jexp, score_im1+_avg_ce_ith, true_sci, yi))
            prev_pop_ijs[which] = (iexp, jexp)

            if yi == self.eos_id:
                # beam items count decrease 1
                if self.ifnorm:
                    self.translations.append(((true_sci / bidx), true_sci, yi, bp, bidx))
                else:
                    self.translations.append((true_sci, true_sci, yi, bp, bidx))
                if len(self.translations) == self.k:
                    # last beam created and finish cube pruning
                    return True
            else:
                yi_emb = self.fn_emb([yi])
                self.beam[bidx].append((true_sci, true_si, yi, yi_emb, bp))

            enough = len(self.beam[bidx]) == (self.k - cnt_transed)
            if enough: return False

            whichsubcub = cube[which]
            # make sure we do not add repeadedly
            if jexp + 1 < each_subcube_colsz[which] and (iexp, jexp + 1) not in ijs_push[which]:
                right = whichsubcub[iexp][jexp + 1]
                ijs_push[which].append((iexp, jexp + 1))
                self.Push_heap(extheap, bidx, right)
            if iexp + 1 < each_subcube_rowsz[which] and (iexp + 1, jexp) not in ijs_push[which]:
                down = whichsubcub[iexp + 1][jexp]
                ijs_push[which].append((iexp + 1, jexp))
                self.Push_heap(extheap, bidx, down)
        return False

    def cube_pruning(self):

        for bidx in range(1, self.maxlen + 1):
            #print bidx

            eq_classes = OrderedDict()
            self.pop_subcube_approx_cache, self.push_subcube_approx_cache, \
                    self.subcube_lines_cache = [], [], []

            self.merge(bidx, eq_classes)

            # create cube and generate next beam from cube
            if self.ifbatch:
                cube = self.create_cube_batch(bidx, eq_classes)
            else:
                cube = self.create_cube(bidx, eq_classes)

            if self.cube_prune(bidx, cube):
                sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                best_sample = sorted_samples[0]
                return back_tracking(self.beam, best_sample, False)

            self.beam[bidx] = sorted(self.beam[bidx], key=lambda tup: tup[0])

        # no early stop, back tracking
        if len(self.translations) == 0:
            best_sample = (self.beam[self.maxlen][0][0],) + \
                self.beam[self.maxlen][0][2:] + (self.maxlen, )
            return back_tracking(self.beam, best_sample, False)
        else:
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            return back_tracking(self.beam, best_sample, False)
