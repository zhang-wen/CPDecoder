import kenlm
#from pytrie import SortedStringTrie as trie
#from pytrie import Trie as trie
from pytrie import StringTrie as trie
import sys
from stream_with_dict import get_tr_stream, ensure_special_tokens
import time
import configurations
import cPickle as pickle
import numpy
from utils import part_sort, debug


# this is to slow !!!  change to another way
def load_language_model(lm, config, trg_vocab, trg_vocab_i2w, ngram, ltrie):
    import kenlm
    m = kenlm.Model(lm)
    sys.stderr.write('use {}-gram langauge model\n'.format(m.order))
    ngram = m.order if ngram > m.order else ngram
    tr_stream = get_tr_stream(**config)
    ltrie = []
    sys.stderr.write('\tload target language model into prefix trie ...')
    idx = 0
    for tr_data in tr_stream.get_epoch_iterator():
        by = tr_data[2]
        idx += 1
        if idx % 5000 == 0:
            logger.debug(idx)
        for y in by:
            y_filter = filter(lambda x: x != 0 and x != config['trg_vocab_size'] - 1, y)
            v_yw = id2w(trg_vocab_i2w, y_filter)
            get_ngram_vocab_prob(m, trg_vocab.keys(), v_yw, ngram, ltrie)
    sys.stderr.write('\tdone')


def id2w(trg_vocab_i2w, y):
    yw = []
    for yy in y:
        yw.append(trg_vocab_i2w[yy])
    debug('id2w... {}'.format(yw))
    return yw


def w2id(trg_vocab, ltrie):
    for t in ltrie:
        for k in t.keys():
            v_ws = k.split()
            v_wids = []
            for w in v_ws:
                v_wids.append(str(trg_vocab[w]))
            vv = t.pop(k)
            new_vv = []
            for v in vv:
                new_vv.append(v[:-1] + (trg_vocab[v[-1]], ))
            t[' '.join(v_wids)] = new_vv

'''
##################################################################

NOTE: (Wen Zhang) get the probability distribution of all words in the vocabulary given one
sentence and ngram, we use prefix trie to restore the distribution for quick query:
    assume:
        vocab = 'x y'
        sentence = 'a b c'
        ngram = 3

    trie[0]:
        {'NULL': logP(x|NULL), logP(y|NULL)}
    trie[1]:
        {'a': [logP(x|a), logP(y|a)],
         'b': [logP(x|b), logP(y|b)],
         'c': [logP(x|c), logP(y|c)]}
    trie[2]:
        {'ab': [logP(x|ab), logP(y|ab)],
         'bc': [logP(x|bc), logP(y|bc)]}

    logP lists are sorted in descending order of log probabilities

##################################################################
'''


def get_ngram_vocab_prob(m, vocab, sent, ngram, ltrie):
    # ngram > 1
    lsent = sent if type(sent) is list else sent.split()

    ldic = []
    # 0, 1, 2, ..., ngram - 1
    for i in xrange(ngram):
        ldic.append({})

    state_in = kenlm.State()
    m.NullContextWrite(state_in)
    # Use <s> as context.  If you don't want <s>, use m.NullContextWrite(state).
    # m.BeginSentenceWrite(ngram_state)
    probs = []
    dist = {}
    for v in vocab:
        state_out = kenlm.State()
        full_score = m.BaseFullScore(state_in, v, state_out)
        # print full_score.log_prob, full_score.ngram_length, full_score.oov
        #probs.append((full_score.log_prob, full_score.ngram_length, full_score.oov, v))
        dist[v] = (full_score.log_prob, full_score.ngram_length, full_score.oov)
    # given 0 word, probs
    # probs.sort(reverse=True)    # lg->sm
    ldic[0]['null'] = trie(dist)

    for wid in range(len(lsent)):
        prev_words = lsent[wid - (ngram - 2) if wid - (ngram - 2) >= 0 else 0:wid + 1]
        gram_m1 = len(prev_words)
        for i in range(1, gram_m1 + 1):
            l_gram_prev = prev_words[-i:]
            s_gram_prev = ' '.join(l_gram_prev)
            # print s_gram_prev
            if not s_gram_prev in ldic[i]:
                probs = []
                state_in = kenlm.State()
                m.NullContextWrite(state_in)
                for w in l_gram_prev:
                    # print w, l_gram_prev
                    ngram_state = kenlm.State()
                    full_score = m.BaseFullScore(state_in, w, ngram_state)
                    # print w
                    # print full_score
                    state_in = ngram_state

                s = time.time()
                for v in vocab:
                    state_out = kenlm.State()
                    full_score = m.BaseFullScore(ngram_state, v, state_out)
                    # print v
                    # print full_score
                    # full_score.ngram_length is the matched ngram length ending with v in
                    # (l_gram_prev + v)
                    #probs.append((full_score.log_prob, full_score.ngram_length, full_score.oov, v))
                    dist[v] = (full_score.log_prob, full_score.ngram_length, full_score.oov)

                print time.time() - s
                print 'add....', len(dist)
                # probs.sort(reverse=True)
                j = 0
                sq = time.time()
                print dist['wonderful']
                print time.time() - sq
                for k, v in dist.iteritems():
                    if j < 10:
                        print k, v
                    j += 1
                ldic[i][s_gram_prev] = trie(dist)

                sq = time.time()
                tdist = trie(dist)
                print 'create trie: ', time.time() - sq

                print tdist.longest_prefix('wandskafjkasdjfas')

                j = 0
                sq = time.time()
                print tdist['wonderful']
                print time.time() - sq
                for k, v in tdist.iteritems():
                    if j < 10:
                        print k, v
                    j += 1

    for i in xrange(ngram):
        ltrie.append(trie(ldic[i]))


def vocab_prob_given_ngram(lm, v_prev_ngram, trg_vocab, trg_vocab_i2w, given=False, wid=True):

    if wid:
        v_prev_ngram = [trg_vocab_i2w[i] for i in v_prev_ngram if i != -1]

    # debug(str(v_prev_ngram))
    logps, wids = [], []
    if given:
        state_in = kenlm.State()
        lm.NullContextWrite(state_in)
        # m.BeginSentenceWrite(state_in)
        for w in v_prev_ngram:
            ngram_state = kenlm.State()
            lm.BaseScore(state_in, w, ngram_state)
            state_in = ngram_state

        for w, idx in trg_vocab.iteritems():
            state_out = kenlm.State()
            log_prob = lm.BaseScore(ngram_state, w, state_out)
            logps.append(log_prob)
            wids.append(idx)

    else:
        for w, idx in trg_vocab.iteritems():
            new_gram = ' '.join(v_prev_ngram + [w])
            log_prob = lm.score(new_gram, bos=False, eos=False)
            logps.append(log_prob)
            wids.append(idx)

    return logps, wids


if __name__ == '__main__':

    vocab = ['i', 'am', 'a', 'good', 'student', '.', 'every', 'has', 'have']

    #test = 'every cloud has a silver lining .'
    test = 'it is good weather today , doesn\'t it ?'
    #m = kenlm.Model('3gram.lc-tok.klm')
    #m = kenlm.Model('input.txt.arpa')

    config = getattr(configurations, 'get_config_cs2en')()

    sys.stderr.write('\tload source and target vocabulary ...\n')
    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    sys.stderr.write('\tvocabulary contains <S>, <UNK> and </S>\n')

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
    sys.stderr.write('\t~done source vocab count: {}, target vocab count: {}\n'.format(
        len(src_vocab), len(trg_vocab)))

    lm = kenlm.Model(
        '/scratch2/wzhang/1.research/4.mt/clause-baesd-mt/lm/train.en.a2b.low.jiutok.o5.binary')

    sys.stderr.write('use {}-gram langauge model\n'.format(lm.order))

    state_in = kenlm.State()
    lm.NullContextWrite(state_in)
    v_prev_ngram_w = ['it', 'is', 'revealed']
    v_prev_ngram_w = ['bolivia', 'holds', 'presidential', 'and']
    v_prev_ngram_w = ['organization', 'of', 'american', 'states']
    v_prev_ngram_w = ['according', 'the']

    probs, wids = vocab_prob_given_ngram(
        lm, v_prev_ngram_w, trg_vocab, trg_vocab_i2w, given=False, wid=False)

    np_probs = numpy.asarray(probs)
    np_wids = numpy.asarray(wids)
    probs_id = part_sort(-np_probs, 10)
    # print probs_id
    print np_probs[probs_id]
    print np_wids[probs_id]
    for i in np_wids[probs_id]:
        print trg_vocab_i2w[i],

    # print probs
    '''
    i = 0
    _k_rank_idx = part_sort(nprobs, 10)
    _k_ith_neg_log_prob = nprobs[_k_rank_idx]
    print _k_ith_neg_log_prob
    for idx in _k_rank_idx:
        print words[idx],
    print
    '''

    '''
    ngram = 4
    ngram = m.order if ngram > m.order else ngram

    ltrie = []
    get_ngram_vocab_prob(m, vocab, test, ngram, ltrie)

    for t in ltrie:
        for i, (k, v) in enumerate(t.iteritems()):
            print k, i
            for vv in v:
                print vv
    '''
