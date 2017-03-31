from __future__ import division

from utils import exeTime, back_tracking, _filter_reidx, init_beam, part_sort, _log, debug
import numpy as np
import time
import sys

DEBUG = False


def debug(s):
    if DEBUG:
        sys.stderr.write('{}\n'.format(s))
        sys.stderr.flush()


class Func(object):

    def __init__(self, lqc, fs, switchs):

        self.lqc = lqc
        self.f_init, self.f_nh, self.f_na, self.f_ns, \
            self.f_mo, self.f_pws, self.f_one, self.f_ce, self.f_next = fs

        self.ifvalid, self.ifbatch, self.ifscore, self.ifnorm, self.ifmv, self.ifwatch_adist, \
            self.merge_way, self.ifapprox_dist, self.ifapprox_att, self.ifadd_lmscore = switchs

    def fn_init(self, np_src):
        self.lqc[0] += 1
        return self.f_init(np_src)

    def fn_nh(self, y_im1, s_im1):
        self.lqc[1] += 1
        return self.f_nh(*[y_im1, s_im1]) if self.ifbatch else self.f_nh(*[[y_im1], s_im1])

    def fn_na(self, c, c_x, hi):
        self.lqc[2] += 1
        return self.f_na(*[c, c_x, hi])

    def fn_ns(self, hi, ai):
        self.lqc[3] += 1
        return self.f_ns(*[hi, ai])

    def fn_mo(self, yemb_im1, ai, hi):
        self.lqc[4] += 1
        return self.f_mo(*[yemb_im1, ai, hi])

    def fn_pws(self, moi, ptv=None):
        self.lqc[5] += 1
        return self.f_pws(moi, ptv) if self.ifmv else self.f_pws(moi)

    def fn_one(self, moi, yi):
        self.lqc[6] += 1
        return self.f_one(*[moi, yi])

    def fn_ce(self, sci):
        self.lqc[7] += 1
        return self.f_ce(sci)

    def fn_next(self, y_im1, ctx, s_im1):
        self.lqc[8] += 1
        return self.f_next(*[y_im1, ctx, s_im1])

class NBS(Func):

    def __init__(self, fs, switchs, bos_id=0, tvcb=None, tvcb_i2w=None,
                 k=10, ptv=None, ln_alpha=0., cp_beta=0.):

        self.lqc = [0] * 11
        super(NBS, self).__init__(self.lqc, fs, switchs)

        self.tvcb = tvcb
        self.tvcb_i2w = tvcb_i2w

        self.bos_id = bos_id
        self.eos_id = len(self.tvcb) - 1

        self.k = k
        self.ptv = ptv

        self.ln_alpha = ln_alpha
        self.cp_beta = cp_beta

    def loss_with_nlcp(self, accum_ces_im1, pi, bp_im1, bp, y_len):
        ys_pi = pi[None, :, :]
        j = bp
        assert(bp_im1 == self.beam[y_len - 1][j][-1])
        for i in reversed(xrange(1, y_len)):
            _, _, p_im1, _, w, backptr = self.beam[i][j]
            ys_pi = np.append(ys_pi, p_im1[None, :, :], axis=0)
            j = backptr
        assert(y_len == len(ys_pi))
        np_ys_pi = ys_pi[::-1]
        ylen = np_ys_pi.shape[0]
        xlen = np_ys_pi.shape[1]
        # print 'np_ys_pi...'
        # print np_ys_pi.shape
        # print np_ys_pi
        y_lp = np.power((5 + y_len), self.ln_alpha) / np.power((5 + 1), self.ln_alpha)
        # print 'y_lp...'
        # print y_lp.shape
        # print y_lp
        x_ysum = np.sum(np_ys_pi, axis=0)
        # print 'x_ysum...'
        # print x_ysum.shape
        # print x_ysum
        logmin = np.log(np.where(x_ysum < 1.0, x_ysum, 1.0))
        # print 'logmin...'
        # print logmin.shape
        # print logmin
        y_cp = self.cp_beta * np.sum(logmin, axis=0)
        return (accum_ces_im1 / y_lp) - y_cp

    def beam_search_trans(self, src_sent):
        if self.ifvalid:
            src = src_sent[0]   # np ndarray
            if len(src_sent) >= 2:
                self.ptv = np.unique(np.array(sorted(src_sent[1])).astype('int64'))
        else:
            src = src_sent
        # subdict set [0,2,6,29999, 333]

        #<type 'list'>
        #[10811, 140, 217, 19, 1047, 482, 29999]
        np_src_sent = np.asarray(src, dtype='int64')
        #<type 'np.ndarray'> (7,)
        #[10811   140   217    19  1047   482 29999]
        if np_src_sent.ndim == 1:  # x (5,)
            # x(5, 1), (slen, batch_size)
            np_src_sent = np_src_sent[:, None]

        src_sent_len = np_src_sent.shape[0]
        self.maxlen = 2 * src_sent_len
        if self.ifbatch:
            best_trans, loss = self.beam_search_comb(np_src_sent)
        else:
            best_trans, loss = self.beam_search(np_src_sent)

        _log('@source[{}], translation(without eos)[{}], maxlen[{}], loss[{}]'.format(
            src_sent_len, len(best_trans), self.maxlen, loss))
        _log('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] ce[{}] next[{}]'.format(*self.lqc[0:9]))

        return _filter_reidx(self.bos_id, self.eos_id, best_trans, self.tvcb_i2w,
                             self.ifmv, self.ptv)

    ##################################################################

    # Wen Zhang: beam search, no batch

    ##################################################################

    @exeTime
    def beam_search(self, np_src_sent):

        self.locrt = [0] * 2
        self.beam = []
        self.translations = []
        maxlen = self.maxlen
        s_init, context, c_x = self.fn_init(np_src_sent)   # np_src_sent (sl, 1), beam==1
        # (1, trg_nhids), (src_len, 1, src_nhids*2)
        init_beam(self.beam, cnt=maxlen, init_state=s_init)

        for i in range(1, maxlen + 1):
            if (i - 1) % 10 == 0:
                debug(str(i - 1))
            cands = []
            for j in xrange(len(self.beam[i - 1])):  # size of last beam
                # (45.32, (beam, trg_nhids), -1, 0)
                accum_loss_im1, accum_im1, _, s_im1, y_im1, bp_im1 = self.beam[i - 1][j]
                #accum_im1, _, s_im1, y_im1, bp_im1 = self.beam[i - 1][j]

                yemb_im1, hi = self.fn_nh(y_im1, s_im1)
                pi, ai = self.fn_na(context, c_x, hi)
                # pi: (src_len, ) sum == 1
                si = self.fn_ns(hi, ai)
                mo = self.fn_mo(yemb_im1, ai, si)
                next_scores = self.fn_pws(mo, self.ptv)

                next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
                next_ces_flat = next_ces.flatten()    # (1,vocsize) -> (vocsize,)
                #ranks_idx_flat = part_sort(next_ces_flat, self.k - len(self.translations))
                ranks_idx_flat = part_sort(next_ces_flat, self.k)
                k_avg_loss_flat = next_ces_flat[ranks_idx_flat]  # -log_p_y_given_x
                # for idx in ranks_idx_flat:
                #    print self.tvcb_i2w[idx],
                # print '\n'

                accum_i = accum_im1 + k_avg_loss_flat
                accum_loss_i = self.loss_with_nlcp(accum_i, pi, bp_im1, j, i)
                cands += [(accum_loss_i[idx], accum_i[idx], pi, si, wid, j)
                          for idx, wid in enumerate(ranks_idx_flat)]
                # cands += [(accum_i[idx], pi, si, wid, j)
                #          for idx, wid in enumerate(ranks_idx_flat)]

            #k_ranks_flat = part_sort(np.asarray(
            #    [cand[0] for cand in cands] + [np.inf]), self.k - len(self.translations))
            k_ranks_flat = part_sort(np.asarray(
                [cand[0] for cand in cands] + [np.inf]), self.k)
            k_sorted_cands = [cands[r] for r in k_ranks_flat]

            for b in k_sorted_cands:
                if b[-2] == self.eos_id:
                    _log('add: {}'.format(((b[0] / i), b[0]) + b[-2:] + (i,)))
                    if self.ifnorm:
                        self.translations.append(((b[0] / i), b[0]) + b[-2:] + (i,))
                    else:
                        self.translations.append((b[0], ) + b[-2:] + (i, ))
                    if len(self.translations) == self.k:
                        # output sentence, early stop, best one in k
                        _log('early stop! see {} samples ending with EOS.'.format(self.k))
                        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                        _log('average location of back pointers [{}/{}={}]'.format(
                            self.locrt[0], self.locrt[1], avg_bp))
                        sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                        best_sample = sorted_samples[0]
                        _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
                        for sample in sorted_samples:  # tuples
                            _log('{}'.format(sample))

                        return back_tracking(self.beam, best_sample)
                else:
                    # should calculate when generate item in current beam
                    self.locrt[0] += (b[-1] + 1)
                    self.locrt[1] += 1
                    self.beam[i].append(b)
            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]:
                debug(b[0:2] + b[-2:])    # do not output state

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        _log('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            _log('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(maxlen))
            best_sample = (self.beam[maxlen][0][0],) + self.beam[maxlen][0][-2:] + (maxlen, )
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample)
        else:
            _log('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                _log('{}'.format(sample))
            return back_tracking(self.beam, best_sample)

    @exeTime
    def beam_search_comb(self, np_src_sent):

        maxlen = self.maxlen
        hyp_scores = np.zeros(1).astype('float32')
        s_init, ctx0, c_x0 = self.fn_init(np_src_sent)   # np_src_sent (sl, 1), beam==1
        init_beam(self.beam, cnt=maxlen, init_state=s_init[0])
        for i in range(1, maxlen + 1):
            # beam search here
            if (i - 1) % 10 == 0:
                debug(str(i - 1))

            prevb = self.beam[i - 1]
            len_prevb = len(prevb)
            cands = []
            # batch states of previous beam
            s_im1 = np.array([b[1] for b in prevb])
            y_im1 = np.array([b[2] for b in prevb])
            # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, len_prevb, 2*src_nhids)
            context = np.tile(ctx0, [len_prevb, 1])
            c_x = np.tile(c_x0, [len_prevb, 1])

            yemb_im1, hi = self.fn_nh(y_im1, s_im1)
            pi, ai = self.fn_na(context, c_x, hi)
            si = self.fn_ns(hi, ai)
            mo = self.fn_mo(yemb_im1, ai, si)
            next_scores = self.fn_pws(mo, self.ptv)

            next_ces = -next_scores if self.ifscore else self.fn_ce(next_scores)
            cand_scores = hyp_scores[:, None] + next_ces
            cand_scores_flat = cand_scores.flatten()
            ranks_flat = part_sort(cand_scores_flat, self.k - len(self.translations))
            voc_size = next_scores.shape[1]
            prevb_id = ranks_flat // voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_scores_flat[ranks_flat]

            for b in zip(costs, si[prevb_id], word_indices, prevb_id):
                if b[2] == self.eos_id:
                    if self.ifnorm:
                        self.translations.append(((b[0] / i), b[0]) + b[2:] + (i, ))
                    else:
                        self.translations.append((b[0], ) + b[2:] + (i,))
                    if len(self.translations) == self.k:
                        # output sentence, early stop, best one in k
                        _log('early stop! see {} samples ending with EOS.'.format(self.k))
                        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
                        _log('average location of back pointers [{}/{}={}]'.format(
                            self.locrt[0], self.locrt[1], avg_bp))
                        sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
                        best_sample = sorted_samples[0]
                        _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
                        for sample in sorted_samples:  # tuples
                            _log('{}'.format(sample))
                        return back_tracking(self.beam, best_sample)
                else:
                    # should calculate when generate item in current beam
                    self.locrt[0] += (b[3] + 1)
                    self.locrt[1] += 1
                    self.beam[i].append(b)
            debug('beam {} ----------------------------'.format(i))
            for b in self.beam[i]:
                debug(b[0:1] + b[2:])    # do not output state
            hyp_scores = np.array([b[0] for b in self.beam[i]])

        # no early stop, back tracking
        avg_bp = format(self.locrt[0] / self.locrt[1], '0.3f')
        _log('average location of back pointers [{}/{}={}]'.format(
            self.locrt[0], self.locrt[1], avg_bp))
        if len(self.translations) == 0:
            _log('no early stop, no candidates ends with EOS, selecting from '
                'len {} candidates, may not end with EOS.'.format(maxlen))
            best_sample = ((self.beam[maxlen][0][0],) + self.beam[maxlen][0][2:] + (maxlen, ))
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            return back_tracking(self.beam, best_sample)
        else:
            _log('no early stop, not enough {} candidates end with EOS, selecting the best '
                'sample ending with EOS from {} samples.'.format(self.k, len(self.translations)))
            sorted_samples = sorted(self.translations, key=lambda tup: tup[0])
            best_sample = sorted_samples[0]
            _log('translation length(with EOS) [{}]'.format(best_sample[-1]))
            for sample in sorted_samples:  # tuples
                _log('{}'.format(sample))
            return back_tracking(self.beam, best_sample)
