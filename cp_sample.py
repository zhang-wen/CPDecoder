import numpy
import copy
import os
import sys
import subprocess
import re
from utils import exeTime, _log, _index2sentence
import collections
from multiprocessing import Process, Queue

from search_mle import mle_trans
from search_naive import ORI
from search_bs import NBS
from search_cp import WCP


class Translator(object):

    def __init__(self, fs, switchs, mode, svcb=None, svcb_i2w=None, tvcb=None, tvcb_i2w=None,
                 bos_idx=0, ngram=3, k=10, thresh=100.0, lm=None, ptv=None, ln_alpha=0., cp_beta=0.):

        self.svcb = svcb
        self.svcb_i2w = svcb_i2w
        self.tvcb = tvcb
        self.tvcb_i2w = tvcb_i2w
        self.mode = mode

        self.ori = ORI(fs, switchs, bos_idx, tvcb, tvcb_i2w, k, ptv, ln_alpha, cp_beta)
        self.nbs = NBS(fs, switchs, bos_idx, tvcb, tvcb_i2w, k, ptv, ln_alpha, cp_beta)
        self.wcp = WCP(fs, switchs, bos_idx, ngram, tvcb, tvcb_i2w, k, thresh, lm, ptv)

    def trans_onesent(self, s):
        avg_merges = 0.
        if self.mode == 0:
            trans = mle_trans(s, fs, switchs, trg_vocab_i2w, k=k)
        elif self.mode == 1:
            self.ori.lqc = [0] * 11
            trans = self.ori.original_trans(s)
        elif self.mode == 2:
            self.nbs.lqc = [0] * 11
            trans = self.nbs.beam_search_trans(s)
        elif self.mode == 3:
            self.wcp.lqc = [0] * 11
            self.wcp.locrt = [0] * 2
            self.wcp.onerow_subcube_cnt = 0
            self.wcp.push_cnt = 0
            self.wcp.pop_cnt = 0
            self.wcp.down_pop_cnt = 0
            self.wcp.right_pop_cnt = 0
            avg_merges, trans = self.wcp.cube_prune_trans(s)
        return avg_merges, trans

    def trans_samples(self, srcs, trgs):
        for index in range(len(srcs)):
            # print 'before filter: '
            # print s[index]
            # [   37   785   600    44   160  4074   152  3737     2   399  1096   170      4     8    29999     0     0     0     0     0     0     0]
            s_filter = filter(lambda x: x != 0, srcs[index])
            _log('\n[{:3}] {}'.format('src', _index2sentence(s_filter, self.svcb_i2w)))
            # ndarray -> list
            # s_filter: [   37   785   600    44   160  4074   152  3737     2   399
            # 1096   170      4 8    29999]
            t_filter = filter(lambda x: x != 0, trgs[index])
            _log('[{:3}] {}'.format('ref', _index2sentence(t_filter, self.tvcb_i2w)))

            _, trans = self.trans_onesent(s_filter)

            _log('[{:3}] {}\n'.format('out', trans))

    @exeTime
    def single_trans_valid(self, x_iter):
        total_trans = []
        total_avg_merge_rate, total_sent_num = 0., 0
        for idx, line in enumerate(x_iter):
            s_filter = filter(lambda x: x != 0, line)
            avg_merges, trans = self.trans_onesent(s_filter)
            total_avg_merge_rate += avg_merges
            total_sent_num += 1
            total_trans.append(trans)
            if numpy.mod(idx + 1, 10) == 0:
                _log('Sample {} Done'.format((idx + 1)))
        _log('Done ...')
        return total_avg_merge_rate / total_sent_num, '\n'.join(total_trans)

    def translate(self, queue, rqueue, pid):

        while True:
            req = queue.get()
            if req == None:
                break

            idx, src = req[0], req[1]
            _log('{}-{}'.format(pid, idx))
            s_filter = filter(lambda x: x != 0, src)
            _, trans = self.trans_onesent(s_filter)

            rqueue.put((idx, trans))

        return

    @exeTime
    def multi_process(self, x_iter, n_process=5):
        queue = Queue()
        rqueue = Queue()
        processes = [None] * n_process
        for pidx in xrange(n_process):
            processes[pidx] = Process(target=self.translate, args=(queue, rqueue, pidx))
            processes[pidx].start()

        def _send_jobs(x_iter):
            for idx, line in enumerate(x_iter):
                # _log(idx, line)
                queue.put((idx, line))
            return idx + 1

        def _finish_processes():
            for pidx in xrange(n_process):
                queue.put(None)

        def _retrieve_jobs(n_samples):
            trans = [None] * n_samples
            for idx in xrange(n_samples):
                resp = rqueue.get()
                trans[resp[0]] = resp[1]
                if numpy.mod(idx + 1, 1) == 0:
                    _log('Sample {}/{} Done'.format((idx + 1), n_samples))
            return trans

        _log('Translating ...')
        n_samples = _send_jobs(x_iter)     # sentence number in source file
        trans_res = _retrieve_jobs(n_samples)
        _finish_processes()
        _log('Done ...')

        return '\n'.join(trans_res)


if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    _log(res)
