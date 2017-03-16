import numpy
import theano
import theano.tensor as T
from collections import OrderedDict
from model import GRU, Lookup_table, DBLSTM, LogisticRegression, BiGRU
from utils import param_init, ln


def _p(pp, name):
    return '%s_%s' % (pp, name)


class Attention(object):

    # s_in is src_nhids*2, t_in is trg_nhids
    def __init__(self, s_in, t_in, prefix='Attention', **kwargs):
        self.params = []
        self.s_in = s_in
        self.t_in = t_in
        self.align_size = t_in  # n_hids -> trg_nhids
        self.prefix = prefix

        self.Wa = param_init().param((self.t_in, self.align_size), name=_p(prefix, 'Wa'))
        # self.v = param_init().param((self.align_size,), init_type='constant',
        # name=_p(prefix, 'v'), scale=0.001)

        self.v = param_init().param((self.align_size,), name=_p(prefix, 'v'))
        self.params += [self.Wa, self.v]

# context, c_mask, h_tm1
# source_ctx -> c:              (src_sent_len, batch_size, src_nhids*2)
# source_mask -> c_mask:    (src_sent_len * batch_size)   actually is the source_mask
# source_x -> c_x:          (src_sent_len, batch_size, trg_nhids)
# cur_hidden -> h1:         init (batch_size, trg_nhids)   calculate from
# previous state
    def apply(self, source_ctx, source_mask, source_x, cur_hidden):
        if source_ctx.ndim != 3 or cur_hidden.ndim != 2:
            raise NotImplementedError

# self.Ws: shape(src_nhids*2, trg_nhids)
# self.Wa: shape(trg_nhids, trg_nhids)
# self.b: shape(trg_nhids,)
# self.v: shape(trg_nhids,)
        align_matrix = T.tanh(source_x + T.dot(cur_hidden, self.Wa)[None, :, :])
# T.tanh(theano.dot(source_ctx, self.Ws): shape(src_sent_len, batch_size, trg_nhids)
# T.dot(cur_hidden, self.Wa):          shape(batch_size, trg_nhids)
# T.dot(cur_hidden, self.Wa)[None,:,:]:shape(1, batch_size, trg_nhids)   # need broadcasting when adding
# align_matrix:     shape(src_sent_len, batch_size, align_size)
        # (src_sent_len, batch_size)
        align = theano.dot(align_matrix, self.v)
# why substract the maximum in one sentence
        # (1, batch_size) if keepdims else (batch_size, )
        align = T.exp(align - align.max(axis=0, keepdims=True))
        if source_mask:     # remove padding words in a batch by multiply 0
            align = align * source_mask
        # after multipling mask, have 0s at ends of each column, we softmax by column
        # align: shape(src_sent_len, batch_size)
        # probability, suming over all words in source sentence, do not
        # consider 0s (padding words)
        # (batch_size, ), shape: array([batch_size])
        normalization = align.sum(axis=0)
        align = align / normalization   # (src_sent_len, batch_size)
        '''
        column becomes probability weights:
        array([[ 0.35776257,  0.60720638,  0.47209168],
               [ 0.64223743,  0.39279362,  0.52790832],
               [ 0.        ,  0.        ,  0.        ]])
        '''
        # align[:, :, None]: shape(src_sent_len, batch_size, 1)
        self.output = (T.shape_padright(align) * source_ctx).sum(axis=0)
        '''
# weighted sum over all words in first dim vector for each elememt in last dim (2*src_nhids, )
# in a batch, for a word (row) of a sentence (column) the state vector source_ctx[row][column] (src_nhids*2, )
# the weight are alignment probabilities in a column of 'align' for source sentence
# multiply same probability align[row][column] which make sense

after multipling mask, have 0s at ends of each column, we softmax by column, get:
array([[ 0.35776257,  0.60720638,  0.47209168,  0.58114407,  0.46887004],
       [ 0.64223743,  0.39279362,  0.52790832,  0.41885593,  0.53112996]])    ->
array([[[ 0.35776257],
        [ 0.60720638],
        [ 0.47209168],
        [ 0.58114407],
        [ 0.46887004]],

       [[ 0.64223743],
        [ 0.39279362],
        [ 0.52790832],
        [ 0.41885593],
        [ 0.53112996]]])	*

(2, 5, 2) (src_sent_len, batch_size, 2*src_nhids)
array([[[ 1,  2],	# should be (2*src_nhids, )
        [ 2,  3],
        [ 3,  4],
        [ 4,  5],
        [ 5,  6]],

       [[ 6,  7],
        [ 7,  8],
        [ 8,  9],
        [ 9, 10],
        [10, 11]]])	=

array([[[ 0.35776257,  0.71552513],
        [ 1.21441275,  1.82161913],
        [ 1.41627503,  1.88836671],
        [ 2.32457628,  2.90572035],
        [ 2.34435022,  2.81322026]],

       [[ 3.8534246 ,  4.49566203],
        [ 2.74955537,  3.14234899],
        [ 4.22326659,  4.75117491],
        [ 3.76970337,  4.1885593 ],
        [ 5.31129957,  5.84242953]]]) *
        '''

        return align, self.output


class Decoder(GRU):  # Decoder is sub-class of GRU
    # n_in and n_hids here are about target sentence, while c_hids are about source sentence
    # n_in -> trgw_embsz, n_hids -> trg_nhids, c_hids -> src_nhids*2

    def __init__(self, n_in, n_hids, c_hids, prefix='Decoder', **kwargs):
        kwargs['c_hids'] = c_hids
        kwargs['max_out'] = False
        kwargs['merge'] = True
# source sentence is contex when we decoding (c_hids == src_nhids*2) the
# result of bi-encoding
        super(Decoder, self).__init__(n_in, n_hids,
                                      with_contex=True, prefix=prefix, **kwargs)
# here W_cz, W_cr, W_ch and W_c_init are all shape (c_hids, n_hids) which
# is (src_nhids*2, trg_nhids) normal distribution
        self.attention_layer = Attention(self.c_hids, self.n_hids)
        self.params.extend(self.attention_layer.params)
        self._init_params2()

    def _init_params2(self):

        f = lambda name: _p(self.prefix, name)
        n_hids = self.n_hids
        size_hh = (n_hids, n_hids)

        self.W_hz2 = param_init().param(size_hh, 'orth', name=f('W_hz2'))
        self.W_hr2 = param_init().param(size_hh, 'orth', name=f('W_hr2'))
        self.W_hh2 = param_init().param(size_hh, 'orth', name=f('W_hh2'))
        self.b_z2 = param_init().param((n_hids,), name=f('b_z2'))
        self.b_r2 = param_init().param((n_hids,), name=f('b_r2'))
        self.b_h2 = param_init().param((n_hids,), name=f('b_h2'))

        self.Ws = param_init().param((self.c_hids, self.n_hids), name=f('Ws'))
        self.bs = param_init().param((self.n_hids,), name=f('bs'))
        self.params += [self.W_hz2, self.W_hr2, self.W_hh2,
                        self.b_z2, self.b_r2, self.b_h2, self.Ws, self.bs]
        '''
        # default False
        if self.ln:
            mul_scale = 1.0
            add_scale = 0.0
            self.g1 = param_init().param((n_hids,), scale=mul_scale, name=f('ln_g1'))
            self.g2 = param_init().param((n_hids,), scale=mul_scale, name=f('ln_g2'))
            self.g3 = param_init().param((n_hids,), scale=mul_scale, name=f('ln_g3'))
            self.g4 = param_init().param((n_hids,), scale=mul_scale, name=f('ln_g4'))
            self.b1 = param_init().param((n_hids,), scale=add_scale, name=f('ln_b1'))
            self.b2 = param_init().param((n_hids,), scale=add_scale, name=f('ln_b2'))
            self.b3 = param_init().param((n_hids,), scale=add_scale, name=f('ln_b3'))
            self.b4 = param_init().param((n_hids,), scale=add_scale, name=f('ln_b4'))
            self.params += [self.g1, self.b1, self.g2, self.b2, self.g3, self.b3, self.g4, self.b4]
        '''

    def init_state(self, ctx, x_mask=None):
        # x_mask is source_mask which is numpy.ndarry
        # context: (src_sent_len, batch_size, src_nhids*2)
        # context[0] could be treated as the encoded context of the first word in
        # source sentence (batch_size, src_nhids*2)
        '''
such as: ctx = numpy.random.rand(3,2,2)
array([[[ 0.59847822,  0.53351651],
        [ 0.29819233,  0.24570638]],

       [[ 0.28542252,  0.34418831],
        [ 0.2749528 ,  0.9848622 ]],

       [[ 0.40692169,  0.73336681],
        [ 0.87122295,  0.91161874]]])
x_mask = numpy.asarray([[1,1],[1,1],[0,1]])  shape(3,2)
array([[1, 1],
       [1, 1],
       [0, 1]])

if ctx * x_mask, then ValueError: operands could not be broadcast together with shapes (3,2,2) (3,2)
we need to broadcase x_mask -> x_mask[:,:,None] shape(3,2,1)
array([[[1],
        [1]],

       [[1],
        [1]],

       [[0],
        [1]]])
then ctx * x_mask[:,:,None] = 
array([[[ 0.59847822,  0.53351651],
        [ 0.29819233,  0.24570638]],

       [[ 0.28542252,  0.34418831],
        [ 0.2749528 ,  0.9848622 ]],

       [[ 0.        ,  0.        ],
        [ 0.87122295,  0.91161874]]])

(ctx * x_mask[:,:,None]).sum(0) = 
array([[ 0.88390073,  0.87770482],  # sum of one sentence
       [ 1.44436808,  2.14218733]])

x_mask.sum(0): array([2, 3])  shape(2,)
x_mask.sum(0)[:,None]:	array([[2],	shape(2,1)  -> array([[2], [2]
                               [3]])                      [3], [3])
if (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0) = 
rray([[ 0.44195037,  0.29256827],
       [ 0.72218404,  0.71406244]])
if (ctx * x_mask[:, :, None]).sum(0) / x_mask.sum(0)[:,None] = 
array([[ 0.44195037,  0.43885241],
       [ 0.48145603,  0.71406244]])
        '''

        if x_mask:
            ctx_mean = (ctx * x_mask[:, :, None]
                        ).sum(0) / x_mask.sum(0)[:, None]
        else:
            ctx_mean = ctx.mean(0)

# ctx_mean: (batch_size, src_nhids*2)
# self.W_c_init: (self.c_hids, self.n_hids) == (src_nhids*2, trg_nhids)
# self.b_init: (self.n_hids,) == (trg_nhids,)
        state = T.tanh(T.dot(ctx_mean, self.W_c_init) + self.b_init)
        '''
array([[ 0.61558374,  0.97485545],
       [ 0.16135434,  0.27840635],	+ numpy.asarray([1,1]) shape(2,)
       [ 0.26672818,  0.0919493 ]])
==
array([[ 1.61558374,  1.97485545],
       [ 1.16135434,  1.27840635],
       [ 1.26672818,  1.0919493 ]])

if
array([[ 0.61558374,  0.97485545],
       [ 0.16135434,  0.27840635],	+ numpy.asarray([1,1,1]) shape(3,)
       [ 0.26672818,  0.0919493 ]])
ValueError: operands could not be broadcast together with shapes (3,2) (3,)
the last dimension must be same, can broadcast
	'''
# state: (batch_size, trg_nhids) + (trg_nhids,) == shape(batch_size, trg_nhids)

        return state

    def _forward(self, state_below, mask_below, context, c_mask):

        if state_below.ndim == 3 and context.ndim == 3:
            n_steps = state_below.shape[0]  # (trg_sent_len - 1) !!!
        else:
            raise NotImplementedError
# (batch_size, trg_nhids) (-1 ~ 1)
        init_state = self.init_state(context, c_mask)
        # init_state: (batch_size, trg_nhids), compress source and get mean of last dimension, and
        # then map use (src_nhids, trg_nhids) to get the initial state
        #_, s, a, ss, als = self.attention_layer.apply(context, c_mask, init_state)
        # context_x is just like a bias or initial in the attiention model,
        # never mind
        context_x = T.dot(context, self.Ws) + self.bs
        non_sequences = [context, c_mask, context_x]
        '''
notice, when function scan is invoked here, following functions are invoked in order:
    *** state, attended = _step_forward_with_attention(state_below[0], mask_below[0], init_state, context, c_mask)
    at time 1: state_1, attended_1 = _step_forward_with_attention(state_below[0], mask_below[0], init_state, context, c_mask)
    at time 2: state_2, attended_2 = _step_forward_with_attention(state_below[1], mask_below[1], state_1, context, c_mask)
    at time 3: state_3, attended_3 = _step_forward_with_attention(state_below[2], mask_below[2], state_2, context, c_mask)
    at time 4: state_4, attended_4 = _step_forward_with_attention(state_below[3], mask_below[3], state_3, context, c_mask)
    ..............
    at time t: state_t, attended_t = _step_forward_with_attention(state_below[t-1], mask_below[t-1], state_(t-1), context, c_mask)
so, the length of state_below and mask_below (which is trg_sent_len-1) can not be less than n_steps 
        '''
# state_below: shape(trg_sent_len-1, batch_size, trgw_embsz)
# mask_below:  shape(trg_sent_len-1, batch_size)
# this is the generation process word by word for target sentence
        rval, updates = theano.scan(self._step_forward_with_attention,
                                    sequences=[state_below, mask_below],
                                    outputs_info=[init_state, None],
                                    non_sequences=non_sequences,
                                    n_steps=n_steps  # trg_sent_len - 1
                                    )
# (no last word </S> for each sentence) (trg_sent_len-1) states based on previous word state in sentences of batch
        self.output = rval[0]
# last dimension is probability weighted sum of words in a sentence,
# corresponds to all states
        self.attended = rval[1]
        # return self.output, self.attended, s, a, ss, als
        # self.output: shape(trg_sent_len-1, batch_size, trg_nhids)
        # self.attended: shape(trg_sent_len-1, batch_size, src_nhids*2)
        return self.output, self.attended

# (state_below, mask_below, init_state, context, c_mask)
# x_t: (batch_size, trgw_embsz)
# x_m: (batch_size,)
# h_tm1: (batch_size, trg_nhids)
# c: (src_sent_len, batch_size, src_nhids*2)
# c_mask: (src_sent_len * batch_size)   actually is the source_mask
# c_x: (src_sent_len, batch_size, trg_nhids) (-1 ~ 1), is just like a bias or initial vector for
# attention
    def _step_forward_with_attention(self, x_t, x_m, h_tm1, c, c_mask, c_x):
        # for sample decoding
        # x_t (y_emb_im1):       shape(k-dead_k, trgw_embsz)   embedding of one target word
        # h_tm1 (cur_state): shape(k-dead_k, trg_nhids)     # k-dead_k is the number of candidates in beam
        # c:                 shape(src_sent_len, k-dead_k, src_nhids*2)
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        # attended = self.attention_layer.apply(c, c_mask, h_tm1)
        # c_z = theano.dot(attended, self.W_cz)
        # c_r = theano.dot(attended, self.W_cr)
        # c_h = theano.dot(attended, self.W_ch)

        # return [self._step_forward_with_context(x_t, x_m, h_tm1, c_z, c_r,
        # c_h), attended]

        # new arc
        h1 = self._step_forward(x_t, x_m, h_tm1)    # RNN
        # h1: (batch_size, trg_nhids)   first hidden state
        # attended = context + source_mask + source_word + first_hidden_state
        _, attended = self.attention_layer.apply(c, c_mask, c_x,  h1)
        # attended:  shape(src_sent_len, batch_size, src_nhids*2)
        # W_cz, W_cr, W_ch: (src_nhids*2, trg_nhids)
        # W_hz2, W_hr2, W_hh2: (trg_nhids, trg_nhids)
        # b_z2, b_r2, b_h2: (trg_nhids, )
        # here is little different from (Bahdanau. etc 2014) which should use h1 as next state and
        # use h_tm1 not h1 to calculate attended
        # should return h1, attended
        h2 = self.state_with_attend(h1, attended, x_m)

        return h2, attended

    def state_with_attend(self, h1, attended, x_m=None):

        # attented: (src_sent_len, batch_size, src_nhids*2)
        _az = theano.dot(attended, self.W_cz) + self.b_z2
        _hz = theano.dot(h1, self.W_hz2)
        if self.ln is not False:
            _az = ln(_az, self.g1, self.b1)
            _hz = ln(_hz, self.g2, self.b2)
        z = T.nnet.sigmoid(_az + _hz)
        # z: (batch_size, trg_nhids)

        _ar = theano.dot(attended, self.W_cr) + self.b_r2
        _hr = theano.dot(h1, self.W_hr2)
        if self.ln is not False:
            _ar = ln(_ar, self.g1, self.b1)
            _hr = ln(_hr, self.g2, self.b2)
        r = T.nnet.sigmoid(_ar + _hr)
        # r: (batch_size, trg_nhids)

        # _ah: (batch_size, trg_nhids)
        _ah = theano.dot(attended, self.W_ch)
        _hh = T.dot(h1, self.W_hh2) + self.b_h2
        if self.ln is not False:
            _ah = ln(_ah, self.g3, self.b3)
            _hh = ln(_hh, self.g4, self.b4)

        h2 = T.tanh(_ah + _hh * r)
        h2 = z * h1 + (1. - z) * h2

        if x_m is not None:
            h2 = x_m[:, None] * h2 + (1. - x_m)[:, None] * h1
        # h2: (batch_size, trg_nhids)
        return h2


# state_below (target): shape(trg_sent_len-1, batch_size, trgw_embsz)
# mask_below (target): shape(trg_sent_len-1, batch_size)
# context (source): shape(src_sent_len, batch_size, src_nhids*2), encode result
# c_mask (source): shape(src_sent_len * batch_size), when decoding, source
# like the context
    def apply(self, state_below, mask_below, context, c_mask):
        hiddens, attended = self._forward(state_below, mask_below, context, c_mask)

# state_below: shape(trg_sent_len-1, batch_size, trgw_embsz)
# hiddens:     shape(trg_sent_len-1, batch_size, trg_nhids)
# attended:    shape(trg_sent_len-1, batch_size, src_nhids*2)
# note: the scan function will remember all privious states
        combine = T.concatenate([state_below, hiddens, attended], axis=2)
# combine:  shape(trg_sent_len-1, batch_size, trgw_embsz+trg_nhids+src_nhids*2)

        # self.W_m: shape(trgw_embsz + trg_nhids + c_hids, n_out*2)
        # self.b_m: shape(n_out*2,)
        if self.max_out:
            merge_out = theano.dot(combine, self.W_m) + self.b_m
            # merge_out: shape(trg_sent_len-1, batch_size, n_out*2)
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1],
                                           merge_out.shape[2] / 2,
                                           2), ndim=4).max(axis=3)

        else:
            merge_out = T.tanh(theano.dot(combine, self.W_m) + self.b_m)

        '''
    such as:  (1, 2, 6)               ->        (1, 2, 3, 2)                  ->      (1, 2, 3)
        [[[ 1, 2, 3, 4, 5, 6],              [[[[1, 2], [3, 4], [4, 5]],         [[[ 2, 4, 5],
          [ 2, 3, 4, 5, 6, 7]]]     ->        [[2, 3], [3, 4], [4, 5]]]]    ->    [ 3, 4, 5]]]
        '''
        # mask_below[:, :, None] -> shape(trg_sent_len-1, batch_size, 1)
        return merge_out * mask_below[:, :, None]

    def merge_out(self, y_emb_im1, s_i, a_i):
        combine = T.concatenate([y_emb_im1, s_i, a_i], axis=1)
        merge_out = theano.dot(combine, self.W_m) + \
            self.b_m  # (k-dead_k, n_out*2)
        if self.max_out:
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1] / 2,
                                           2), ndim=3).max(axis=2)
        else:
            merge_out = T.tanh(merge_out)
        return merge_out


class Translate(object):

    def __init__(self,
                 enc_nhids=1000,
                 dec_nhids=1000,
                 enc_embed=620,
                 dec_embed=620,
                 src_vocab_size=30000,
                 trg_vocab_size=30000,
                 **kwargs):
        self.lr_in = kwargs.get('n_out', dec_nhids)

        self.src_lookup_table = Lookup_table(
            enc_embed, src_vocab_size, prefix='src_lookup_table')
        self.trg_lookup_table = Lookup_table(
            dec_embed, trg_vocab_size, prefix='trg_lookup_table')
        self.encoder = BiGRU(enc_embed, enc_nhids, **kwargs)
# src_nhids*2 corresponds the last dimension of encoded state
        self.decoder = Decoder(dec_embed, dec_nhids,
                               c_hids=enc_nhids * 2, **kwargs)
        # the output size of decoder should be same with lr_in if no n_out
        # defined
        self.logistic = LogisticRegression(
            self.lr_in, trg_vocab_size, prefix='logistic', **kwargs)
        self.params = self.src_lookup_table.params + self.trg_lookup_table.params + \
            self.encoder.params + self.decoder.params + self.logistic.params
        self.tparams = OrderedDict([(param.name, param)
                                    for param in self.params])
        self.use_mv = kwargs.get('use_mv', 0)

    def apply(self, source, source_mask, target, target_mask, v_part=None, v_true=None, **kwargs):
        # sbelow and tbelow are 3-D matrix, sbelow[i][j] (tbelow[i][j]) are embeddings of the i^{th} word in the j^{th} sentence in batch
        # here, source and source_mask: shape(src_sent_len * batch_size)
        # target and target_mask: shape(trg_sent_len * batch_size)
        # and their type are all theano.tensor.var.TensorVariable (numpy.ndarray)
        # (40,28,620) = (src_sent_len, batch_size, srcw_embsz)
        sbelow = self.src_lookup_table.apply(source)
        # the shape is different from source, (trg_sent_len-1, batch_size,
        # trgw_embsz)
        tbelow = self.trg_lookup_table.apply_zero_pad(target)

        # (src_sent_len, batch_size, src_nhids*2): bidirectional encode source sentence
        s_rep = self.encoder.apply(sbelow, source_mask)
        # remove the last word which is '</S>' of each sentence in a batch, the padding words are alse </S> 29999
        # tbelow[:-1] -> shape(trg_sent_len-1, batch_size, trgw_embsz)
        # target_mask[:-1] -> shape(trg_sent_len-1, batch_size)
        # hiddens, s, a, ss, als = self.decoder.apply(tbelow[:-1], target_mask[:-1], s_rep, source_mask)
        hiddens = self.decoder.apply(tbelow, target_mask, s_rep, source_mask)
        # hiddens from decoder: shape(trg_sent_len-1, batch_size, n_out)
        # (padding words all 0)
        self.mean_cost, self.mean_abs_log_norm = self.logistic.cost(
            hiddens, target, target_mask, v_part, v_true)

        # cost_matrix: shape((trg_sent_len-1), batch_size), here the trg_sent_len corresponds to this batch,
        # trg_sent_len may differ between different batches
        # cost_matrix.sum(): sum of all the elements in cost_matrix
        # target_mask[1]: the sentences number in a batch
        # so, cost_matrix.sum()/target_mask.shape[1] is actually the average cross
        # entropy per sentence in a batch

    '''
    y_emb_im1: (trgw_embsz,)
    t_stat_im1: (batch_size, trg_nhids)
    ctx: (src_sent_len, batch_size, src_nhids*2)
    c_x: (src_sent_len, batch_size, trg_nhids)
    '''

    def build_sample(self):
        x = T.matrix('x', dtype='int64')
        sbelow = self.src_lookup_table.apply(x)
        mask = T.alloc(numpy.float32(1.), sbelow.shape[0], sbelow.shape[1])
        # (src_sent_len, batch_size, src_nhids*2) batch_size == 1 for decoding
        ctx = self.encoder.apply(sbelow, mask)
        # self.decoder.Ws: (src_nhids*2, trg_nhids)
        # self.decocer.bs: (trg_nhids, )
        # (src_sent_len, batch_size, trg_nhids) (-1 ~ 1)
        # as long as ctx is inputed as parameter, no need c_x, it will be
        # calculated... do not worry
        c_x = T.dot(ctx, self.decoder.Ws) + self.decoder.bs
        # init_state: (batch_size, trg_nhids)
        init_state = self.decoder.init_state(ctx)   # no mask here, because no batch
        f_init = theano.function([x], [init_state, ctx, c_x], name='f_init')

        #--------------------------------------------------------------
        y_im1 = T.vector('y_sampler', dtype='int64')
        t_stat_im1 = T.matrix('t_stat_im1', dtype='float32')

        #--------------------------------------------------------------
        # get next state h1: h_i = rnn(y_{i-1}, s_{i-1})
        # y_emb_im1: embedding of one target word, shape(1, trgw_embsz)
        y_emb_im1 = self.trg_lookup_table.index(y_im1)
        hi = self.decoder._step_forward(x_t=y_emb_im1, x_m=None, h_tm1=t_stat_im1)
        f_nh = theano.function([y_im1, t_stat_im1], [y_emb_im1, hi], name='f_nh')

        #--------------------------------------------------------------
        t_hi = T.matrix('t_hi', dtype='float32')
        t_ctx = T.tensor3('t_ctx', dtype='float32')
        t_c_x = T.tensor3('t_c_x', dtype='float32')
        # next attention: a_i = a(h_i, c_i), c_i is actually do not change ...
        pi, ai = self.decoder.attention_layer.apply(
            source_ctx=t_ctx, source_mask=None, source_x=t_c_x, cur_hidden=t_hi)
        f_na = theano.function([t_ctx, t_c_x, t_hi], [pi, ai], name='f_na')

        #--------------------------------------------------------------
        # get next final state, s_i = f(h_i<=(y_{i-1} and s_{i-1}), y_{i-1},
        # c_i)
        t_ai = T.matrix('t_ai', dtype='float32')
        ns = self.decoder.state_with_attend(h1=t_hi, attended=t_ai)
        f_ns = theano.function([t_hi, t_ai], ns, name='f_ns')

        #--------------------------------------------------------------
        # merge_out = g(y_{i-1}, s_i, a_i)
        t_si = T.matrix('t_si', dtype='float32')
        t_yemb_im1 = T.matrix('t_yemb_im1', dtype='float32')
        merge_out = self.decoder.merge_out(
            y_emb_im1=t_yemb_im1, s_i=t_si, a_i=t_ai)
        f_mo = theano.function([t_yemb_im1, t_ai, t_si],
                               merge_out, name='f_mo')

        #--------------------------------------------------------------
        # get model score of the whole vocab: nonlinear(merge_out)
        t_mo = T.matrix('t_mo', dtype='float32')
        if self.use_mv:
            ptv = T.vector('ptv', dtype='int64')
            ptv_ins = [t_mo, ptv]
            ptv_ous = self.logistic.apply_score(t_mo, ptv, drop=True)
        else:
            ptv_ins = [t_mo]
            ptv_ous = self.logistic.apply_score(t_mo, drop=True)
        f_pws = theano.function(ptv_ins, ptv_ous, name='f_pws')

        #--------------------------------------------------------------
        # no need to use the whole vocabulary, vocabulary manipulation
        # if use T.ivector(), this slice will be very slow on cpu, i do not
        # know why
        y = T.wscalar('y')
        # get part model score slice: nonlinear(merge_out)[part]
        f_one = theano.function(
            [t_mo, y], self.logistic.apply_score_one(t_mo, y), name='f_one')

        #--------------------------------------------------------------
        # distribution over target vocab: softmax(energy)
        t_pws = T.matrix('t_pws', dtype='float32')
        #self.logistic.apply_softmax(t_pws)
        self.logistic.softmax(t_pws)
        f_ce = theano.function([t_pws], self.logistic.ce_p_y_give_x, name='f_ce')
        # next_w(y_emb_im1):    (k-dead_k,)  the last word id of each translate candidate in beam
        # ctx:  (src_sent_len, live_k, src_nhids*2)
        # t_stat_im1:           shape(k-dead_k, trg_nhids)
        # probs:  shape(k-dead_k, trg_vocab_size)
        return [f_init, f_nh, f_na, f_ns, f_mo, f_pws, f_one, f_ce]

    def savez(self, filename):
        params_value = OrderedDict([(kk, value.get_value())
                                    for kk, value in self.tparams.iteritems()])
        numpy.savez(filename, **params_value)

    def load(self, filename):  # change all weights by file
        params_value = numpy.load(filename)
        assert len(params_value.files) == len(self.tparams)
        for key, value in self.tparams.iteritems():
                        # type(value) theano.tensor.sharedvar.TensorSharedVariable
                        # params_value[key] is numpy.ndarray
                        # we set the shared variable as the numpy array
            value.set_value(params_value[key])
        '''
        type(params_value['logistic_W0']: numpy.ndarray (512, 30000)
array([[-0.00096034, -0.0392303 , -0.07458289, ..., -0.00285031,
         0.03942127, -0.03161906],
       [-0.03706803, -0.06445373, -0.00836279, ..., -0.01915432,
        -0.00247126,  0.17407075],
       [-0.00102945,  0.03983303, -0.00801838, ..., -0.02834764,
         0.02834882, -0.07769781],
       ...,
       [ 0.01267207,  0.07802714, -0.02748013, ...,  0.0485581 ,
        -0.00657458,  0.07204553],
       [ 0.01089897,  0.06406539, -0.04804269, ..., -0.03247456,
         0.04343275, -0.14596273],
       [ 0.01474529,  0.02925147,  0.01569422, ...,  0.01673588,
        -0.02202134,  0.19972666]], dtype=float32)
        '''

    def load2numpy(self, filename):  # change all weights by file
        params_value = numpy.load(filename)
        assert len(params_value.files) == len(self.tparams)
        return params_value
