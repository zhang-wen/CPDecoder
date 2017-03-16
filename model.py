import theano
import theano.tensor as T
import numpy
from utils import param_init, repeat_x, ln, _log, debug


def _p(pp, name):
    return '%s_%s' % (pp, name)


def _dropout_from_layer(layer, p):
    """p is the probablity of dropping a unit
    """
    rng = numpy.random.RandomState(1234)
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(1234))
    # p=1-p because 1's indicate keep and p is prob of dropping
    # throw coins n times, how many positives, generate 0/1 randomly
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # generate 1 by probablity 1-p, generate 0 by probablity p which is dropped out
    '''
    such as:
        layer: [[1, 2, 3],          mask: [[1, 0, 1],
                [3, 5, 4]]                 [1, 0, 1]]
        output: [[1, 0, 3],
                 [3, 0, 4]]
        each time invoke mask, the value of mask will change ....
    '''
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class LogisticRegression(object):

    def __init__(self, n_in, lr_out, prefix='logist', **kwargs):

        self.n_in = n_in  # n_out: 512    the last layer of decoder merge out
        self.lr_out = lr_out  # 30000  trg_vocab_size which is the predicted target vocabulary size
        # W0: shape(n_out, 30000)
        # b:  shape(30000,)
        self.W0 = param_init().param((n_in, lr_out), name=_p(prefix, 'W0'))
        # self.b = param_init().param((lr_out, ), name=_p(prefix, 'b'))
        self.b = param_init().param((lr_out, ), name=_p(prefix, 'b'), scale=numpy.log(1. / lr_out))
        self.params = [self.W0, self.b]
        self.drop_rate = kwargs.pop('dropout', 0.5)
        self.alpha = kwargs.pop('alpha', 0.0)
        self.use_mv = kwargs.pop('use_mv', 0)

    '''

    input: shape(k-dead_k, n_out) if no n_out para, n_out is trg_nhids, when decoding
            or (trg_src_sent_len, batch_size, n_out) when training

    logit: shape(trg_src_sent_len, batch_size, trg_vocab_size)
        = (trg_src_sent_len, batch_size, n_out) * (n_out, trg_vocab_size)

	array([[[  1.,   2.,   3.],
            [  3.,   4.,   5.],
            [  2.,   2.,   2.]],

           [[  3.,  21.,   2.],
            [  4.,   2.,   2.],
            [ 11.,   7.,   9.]]])   (2, 3, 3,)

    after reshape: (2 * 3, 3)
	array([[  1.,   2.,   3.],
           [  3.,   4.,   5.],
           [  2.,   2.,   2.],
           [  3.,  21.,   2.],
           [  4.,   2.,   2.],
           [ 11.,   7.,   9.]])

    because softmax can only be use for (1-d or 2-d tensor of floats), so we should reshape
    logit reshape: (k-dead_k, vocab_size) or (trg_src_sent_len * batch_size, trg_vocab_size)
    p_y_given_x: shape(trg_src_sent_len * batch_size, trg_vocab_size)
    elements in the last dimension becomes probablities by using softmax
    each element p_y_given_x represents the probablity of each target word (word i in each batch j)
    30,000 columns huge!!! trg_vocab_size huge, softmax here is very slow

    after softmax:

        array([[  9.00305732e-02,   2.44728471e-01,   6.65240956e-01],
               [  9.00305732e-02,   2.44728471e-01,   6.65240956e-01],
               [  3.33333333e-01,   3.33333333e-01,   3.33333333e-01],
               [  1.52299794e-08,   9.99999979e-01,   5.60279632e-09],
               [  7.86986042e-01,   1.06506979e-01,   1.06506979e-01],
               [  8.66813332e-01,   1.58762400e-02,   1.17310428e-01]])

    speed here ... just split calculations
    when decoding, we just need to give the index of subvocab in large vocab
    like [0, 1, 3, 23]

     will this speed the update ?
    '''

    def apply_score_one(self, input, i):
        # input: (trg_sent_len, batch_size, n_out) for training
        #        (1 or beamsize, n_out) for decoding
        # W0:    (n_out, voc_size)
        if self.drop_rate > 0.:
            input = input * (1 - self.drop_rate)
        # self.b need to broadcasting
        logit = theano.dot(input, self.W0[:, i]) + self.b[i]
        # logit.shape: array([1]) or array([80]) if input.shape[0]==80
        if logit.ndim == 3:
            logit = logit.reshape([logit.shape[0] * logit.shape[1], logit.shape[2]])
        # when training, the logit.ndim is 3 (trg_sent_len, batch_size, voc_size)
        # when decoding, the logit.ndim is 2 (1 or beam_size, voc_size)
        return logit

    def apply_score(self, input, v_part=None, drop=False):
        # input: (trg_sent_len, batch_size, n_out) for training
        #        (1 or beamsize, n_out) for decoding
        # W0:    (n_out, voc_size)
        if drop is True and self.drop_rate > 0.:
            input = input * (1 - self.drop_rate)
        if self.use_mv:
            _log(" using batch-level voc. logit[v]")
            W_tran = T.transpose(self.W0)
            # self.b need to broadcasting
            logit = theano.dot(input, T.transpose(W_tran[v_part])) + self.b[v_part]
            # logit.shape: array([1, 3]) or array([80, 3]) if input.shape[0]==80
        else:
            _log(" using full output voc.")
            logit = theano.dot(input, self.W0) + self.b
        if logit.ndim == 3:
            logit = logit.reshape([logit.shape[0] * logit.shape[1], logit.shape[2]])
        # when training, the logit.ndim is 3 (trg_sent_len, batch_size, voc_size)
        # when decoding, the logit.ndim is 2 (1 or beam_size, voc_size)
        return logit

    def apply_softmax(self, x):
        x_exp = T.exp(x)
        sum_exp = T.sum(x_exp, axis=1, keepdims=True)
        self.log_norm = T.log(sum_exp)
        self.p_y_given_x = x_exp / sum_exp
        self.log_p_y_given_x = x - self.log_norm
        self.ce_p_y_give_x = -self.log_p_y_given_x

    def softmax(self, x):
        x_max = T.max(x, axis=1, keepdims=True)  # take max for numerical stability
        self.log_norm = T.log(T.sum(T.exp(x - x_max), axis=1, keepdims=True)) + x_max
        self.log_p_y_given_x = x - self.log_norm
        self.ce_p_y_give_x = -self.log_p_y_given_x

    def cost(self, input, targets, mask=None, v_part=None, v_true=None):

        if self.drop_rate > 0.:
            input = _dropout_from_layer(input, self.drop_rate)

        logit = self.apply_score(input, v_part)
        self.softmax(logit)

        golden_idx = targets.flatten()  # must same with logit.shape[0]
        # oh, the error may because the targets is (y_maxlen, batch),
        # so v_true should also be (y_maxlen, batch) !!!
        if self.use_mv:
            _log(" building y flat idx with batch-level voc. ")
            golden_flat_idx = T.arange(golden_idx.shape[0]) * v_part.shape[0] + v_true.flatten()
        else:
            golden_flat_idx = T.arange(golden_idx.shape[0]) * self.lr_out + golden_idx

        gold_cost_flat = self.ce_p_y_give_x.flatten()[golden_flat_idx]
        log_norm_flat = self.log_norm.flatten()
        shape = [targets.shape[0], targets.shape[1]]
        if mask is not None:
            gold_cost_shape = gold_cost_flat.reshape(shape) * mask
            log_norm_shape = log_norm_flat.reshape(shape) * mask
        norm_cost_shape = gold_cost_shape + self.alpha * (log_norm_shape ** 2)
        # to observe how much we compressed log |Z(x)|
        return T.mean(norm_cost_shape), T.mean(T.abs_(log_norm_shape))

    def errors(self, y):
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()
        return T.sum(T.neq(y, y_pred))


class GRU(object):

    def __init__(self, n_in, n_hids, ln=False, with_contex=False, merge=True,
                 max_out=True, prefix='GRU', **kwargs):
        self.n_in = n_in        # the units number of input layer (embsize)
        self.n_hids = n_hids    # the units number of hidden layer
        self.ln = ln
        debug('ln: {}'.format(self.ln))
        self.with_contex = with_contex  # whether we use the context
        if self.with_contex:
            # the units number of context embedding layer
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self.prefix = prefix
        self.merge = merge
        if self.merge:
            self.n_out = kwargs.pop('n_out', n_hids)  # default units number of the last layer
        self.max_out = max_out  # default True

        self._init_params()

    def _init_params(self):

        f = lambda name: _p(self.prefix, name)  # return 'GRU_' + parameters name

        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        # following three are parameters matrix from input layer to hidden layer:
        # generate numpy.ndarray by normal distribution
        self.W_xz = param_init().param(size_xh, name=f('W_xz'))
        self.W_xr = param_init().param(size_xh, name=f('W_xr'))
        self.W_xh = param_init().param(size_xh, name=f('W_xh'))

        # following three are parameters matrix from hidden layer to hidden layer:
        # generate numpy.ndarray by standard normal distribution with qr
        # factorization
        self.W_hz = param_init().param(size_hh, 'orth', name=f('W_hz'))
        self.W_hr = param_init().param(size_hh, 'orth', name=f('W_hr'))
        self.W_hh = param_init().param(size_hh, 'orth', name=f('W_hh'))

        # following three are bias vector of hidden layer: generate by normal distribution
        self.b_z = param_init().param((n_hids,), name=f('b_z'))
        self.b_r = param_init().param((n_hids,), name=f('b_r'))
        self.b_h = param_init().param((n_hids,), name=f('b_h'))

# just put all this parameters matrix (numpy.ndarray) into a list
        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_contex:    # default False
            size_ch = (self.c_hids, self.n_hids)    # (src_nhids*2, trg_nhids)
# following there are parameters matrix from context hidden layer to hidden layer
            size_ch_ini = (self.c_hids, self.n_hids)
            self.W_cz = param_init().param(size_ch, name=f('W_cz'))
            self.W_cr = param_init().param(size_ch, name=f('W_cr'))
            self.W_ch = param_init().param(size_ch, name=f('W_ch'))
            self.W_c_init = param_init().param(size_ch_ini, name=f('W_c_init'))
            self.b_init = param_init().param((self.n_hids,), name=f('b_init'))

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]  # just put several matrix together

            msize = self.n_in + self.n_hids + self.c_hids
        else:
            msize = self.n_in + self.n_hids

        if self.merge:  # default True
            osize = self.n_out  # default is units number of hidden layer (n_hids == trg_nhids)
            if self.max_out:    # default True, need change here, because it is same w/o max_out
                self.W_m = param_init().param((msize, osize * 2), name=_p(self.prefix, 'W_m'))
                self.b_m = param_init().param((osize * 2,), name=_p(self.prefix, 'b_m'))
                self.params += [self.W_m, self.b_m]
            else:
                self.W_m = param_init().param((msize, osize), name=_p(self.prefix, 'W_m'))
                self.b_m = param_init().param((osize,), name=_p(self.prefix, 'b_m'))
                self.params += [self.W_m, self.b_m]

        # default False
        if self.ln:
            mul_scale = 1.0
            add_scale = 0.0
            self.g1 = param_init().param((n_hids,), scale=mul_scale, name=_p(self.prefix, 'ln_g1'))
            self.g2 = param_init().param((n_hids,), scale=mul_scale, name=_p(self.prefix, 'ln_g2'))
            self.g3 = param_init().param((n_hids,), scale=mul_scale, name=_p(self.prefix, 'ln_g3'))
            self.g4 = param_init().param((n_hids,), scale=mul_scale, name=_p(self.prefix, 'ln_g4'))
            self.b1 = param_init().param((n_hids,), scale=add_scale, name=_p(self.prefix, 'ln_b1'))
            self.b2 = param_init().param((n_hids,), scale=add_scale, name=_p(self.prefix, 'ln_b2'))
            self.b3 = param_init().param((n_hids,), scale=add_scale, name=_p(self.prefix, 'ln_b3'))
            self.b4 = param_init().param((n_hids,), scale=add_scale, name=_p(self.prefix, 'ln_b4'))
            self.params += [self.g1, self.b1, self.g2, self.b2, self.g3, self.b3, self.g4, self.b4]
            if self.with_contex:
                self.gcz = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix, 'ln_gcz'))
                self.bcz = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix,
                                                                                       'ln_bcz'))
                self.gcr = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix,
                                                                                       'ln_gcr'))
                self.bcr = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix,
                                                                                       'ln_bcr'))
                self.gch = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix,
                                                                                       'ln_gch'))
                self.bch = param_init().param((self.n_hids,), scale=mul_scale, name=_p(self.prefix,
                                                                                       'ln_bch'))

    '''
# x_t: input at time t (batch_size, trgw_embsz)
# x_m: mask of x_t (batch_size, )
# h_tm1: (batch_size, trg_nhids) previous state
# c_z, c_r, c_h: (batch_size, trg_nhids) contex of the rnn
    def _step_forward_with_context(self, x_t, x_m, h_tm1, c_z, c_r, c_h):
        _xz = T.dot(x_t, self.W_xz) + self.b_z
        _hz = T.dot(h_tm1, self.W_hz)
        if self.ln:
            _xz = ln(_xz, self.g1, self.b1)
            _hz = ln(_hz, self.g2, self.b2)
        z_t = T.nnet.sigmoid(_xz + _hz + c_z)

        _xr_c = T.dot(x_t, self.W_xr) + self.b_r
        _hr = T.dot(h_tm1, self.W_hr)
        if self.ln:
            _xr = ln(_xr, self.g1, self.b1)
            _hr = ln(_hr, self.g2, self.b2)
        r_t = T.nnet.sigmoid(_xr + _hr + c_r)

        _xh = T.dot(x_t, self.W_xh) + self.b_h
        _hh = T.dot(h_tm1, self.W_hh)
        if self.ln:
            _xh = ln(_xh, self.g3, self.b3)
            _hh = ln(_hh, self.g4, self.b4)

        can_h_t = T.tanh(_xh + _hh * r_t + c_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        if x_m is not None:
            h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        return h_t
    '''

    def _step_forward(self, x_t, x_m, h_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        when decoding:
            x_t: embedding of a word in target sentence (1, trgw_embsz)
            h_tm1: hidden state (batch_size, trg_nhids)
            result h_t: next hidden state (batch_size, trg_nhids)
        '''
        _xz = T.dot(x_t, self.W_xz) + self.b_z
        _hz = T.dot(h_tm1, self.W_hz)
        if self.ln is not False:
            _xz = ln(_xz, self.g1, self.b1)
            _hz = ln(_hz, self.g2, self.b2)
        z_t = T.nnet.sigmoid(_xz + _hz)

        _xr = T.dot(x_t, self.W_xr) + self.b_r
        _hr = T.dot(h_tm1, self.W_hr)
        if self.ln is not False:
            _xr = ln(_xr, self.g1, self.b1)
            _hr = ln(_hr, self.g2, self.b2)
        r_t = T.nnet.sigmoid(_xr + _hr)

        _xh = T.dot(x_t, self.W_xh) + self.b_h
        _hh = T.dot(h_tm1, self.W_hh)
        if self.ln is not False:
            _xh = ln(_xh, self.g3, self.b3)
            _hh = ln(_hh, self.g4, self.b4)

        can_h_t = T.tanh(_xh + _hh * r_t)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t
        # just because this, look for the reason for 6 hours, when it is a little different from training, error...
        #h_t = z_t * h_tm1 + (1. - z_t) * can_h_t

        if x_m is not None:
            h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        return h_t

    # state_below: TensorType(float64, 3D)
    # mask_below: TensorType(float64, matrix)
    # they are both theano.tensor.var.TensorVariable
    def _forward(self, state_below, mask_below=None, init_state=None, context=None):
        if state_below.ndim == 3:  # state_below is a 3-d matrix
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError

# state_below:(src_sent_len,batch_size,embsize),
# mask_below:(src_sent_len,batch_size) 0-1 matrix (padding)
        if mask_below:
            inps = [state_below, mask_below]
            if self.with_contex:
                fn = self._step_forward_with_context
            else:
                fn = self._step_forward
        else:
            inps = [state_below]
            if self.with_contex:
                fn = lambda x1, x2, x3, x4, x5: self._step_forward_with_context(
                    x1, None, x2, x3, x4, x5)
            else:
                fn = lambda x1, x2: self._step_forward(x1, None, x2)

        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init) + self.b_init)
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            if self.ln:
                c_z = ln(c_z, self.gcz + self.bcz)
                c_r = ln(c_r, self.gcr + self.bcr)
                c_h = ln(c_h, self.gch + self.bch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(fn,
                                        sequences=inps,
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                # init_state = T.unbroadcast(T.alloc(0., batch_size, self.n_hids), 0)
            rval, updates = theano.scan(fn,
                                        sequences=inps,
                                        outputs_info=[init_state],
                                        n_steps=n_steps
                                        )
        self.output = rval
# if change like this, it only return the hidden state of the last word in the sentence
        return self.output

# merge the hidden state matrix with the state_below matrix at the last dimension
    def _merge_out(self, state_below, mask_below=None, context=None):
        # state_below:(src_sent_len,batch_size,embsize),
        # mask_below:(src_sent_len,batch_size) 0-1 matrix (padding)
        hiddens = self._forward(state_below, mask_below=None, context=context)
# hiddens: all hidden state of all words for a sentence in a batch (batch_size sentences)
# hiddens: shape (src_sent_len, batch_size, n_hids)
        if self.with_contex:  # default False
            assert context is not None
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)  # n_times * context.shape[0] * context.shape[1]
# combine tensor at the last dimension, such as T.concatenate([2,3,4], [2,3,6], axis=2) = [2,3,10]
            # the size of last dimension becomes (embsize+n_hids)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)
        else:
            combine = T.concatenate([state_below, hiddens], axis=2)

        # combine: (src_sent_len,batch_size,embsize+n_hids)
        # W_m: (msize, osize*2), if having context, msize is (embsize+n_hids+c_hids)
        # else if not having contex, msize is (embsize+n_hids)
        # if no parameter 'n_out', osize is n_hids; else osize is n_out
        # here, W_m: (embsize+n_hids, n_hids*2), normal initial; b_m: (n_hids*2,
        # ), contant 0 initial
        if self.max_out:
            # (src_sent_len,batch_size,n_hids*2) + (n_hids*2, )
            merge_out = theano.dot(combine, self.W_m) + self.b_m
            #(n_hids*2, ) need to be broadcasting number (src_sent_len,batch_size)
# split the the last dimension (,n_hids*2) of merge_out into (n_hids, 2),
# then get the max of the last dimension (each line in (n_hids, 2))
            merge_out = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1],
                                           merge_out.shape[2] / 2,
                                           2), ndim=4).max(axis=3)
            # merge_out: (src_sent_len, batch_size, n_hids)
        else:
            # (src_sent_len, batch_size, n_hids*2): -1~1
            merge_out = T.tanh(theano.dot(combine, self.W_m) + self.b_m)
        if mask_below:
            # (src_sent_len, batch_size, n_hids)
            # mask_below[:, :, None] -> (src_sent_len,batch_size,1)
            merge_out = merge_out * mask_below[:, :, None]
        # if no words for a sentence in a batch, the corresponding [sent_no, batch_no, *] are all 0s
        return merge_out

# state_below:(src_sent_len,batch_size,embsize), mask_below:(src_sent_len,batch_size)
    def apply(self, state_below, mask_below, context=None):
        if self.merge:  # default True
            return self._merge_out(state_below, mask_below, context)
        else:
            return self._forward(state_below, mask_below, context)


class Lookup_table(object):
    # embsize: the embedding size of a word (620)
    # vocab_size: the vocabulary size (30000)

    def __init__(self, embsize, vocab_size, prefix='Lookup_table'):
        # '%s_%s' % (pp, name)   # the name of self.W is 'Lookup_table_embed': self.W
        self.W = param_init().param((vocab_size, embsize), name=_p(prefix, 'embed'))
        # the type of self.W is theano.tensor.sharedvar.TensorSharedVariable: type(self.W)
        # self.W.shape.eval(): (array([30000,   620]), self.W.type: TensorType(float64, matrix)
        # self.W.dtype: 'float64'
        self.params = [self.W]
        self.vocab_size = vocab_size
        self.embsize = embsize

# indices is source or target, whose shape is (src_sent_len * batch_size)
# the type of source or target is theano.tensor.var.TensorVariable (numpy.ndarray)
# indices: (src_sent_len, batch_size), so
# [indices.shape[i] for i in range(indices.ndim)] is [src_sent_len, batch_size]
# outshape is [src_sent_len, batch_size, embsize]
    def apply(self, indices):
        outshape = [indices.shape[i] for i
                    in range(indices.ndim)] + [self.embsize]
        # index the embedding of each word from the dictionary embedding
        # W:    vocab_size row
        #       embsize column
        # W[indices.flatten()]: (indices.row * indices.col) column
        #                       embsize row
        # outshape: [indices.row, indices.col, embsize]
        # return a 3-D matrix, result[i][j] represents a embedding of some word

        #   if indices is array([[1, 1],
        #                        [2, 2],
        #                        [3, 3]])
        #   indices.flatten() is array([1, 1, 2, 2, 3, 3])
        # if self.W.eval() is array([[ 0.00471435, -0.01190976],
        #                            [ 0.01432707, -0.00312652],
        #                            [-0.00720589,  0.00887163],
        #                            [ 0.00859588, -0.00636524]]) shape:(4,2) = (vocab_size,embsize)
        # self.W[indices.flatten()] is array([[ 0.01432707, -0.00312652],
        #                                     [ 0.01432707, -0.00312652],
        #                                     [-0.00720589,  0.00887163],
        #                                     [-0.00720589,  0.00887163],
        #                                     [ 0.00859588, -0.00636524],
        #                                     [ 0.00859588, -0.00636524]]) shape:(6,2)=(src_sent_len*batch_size, embsize)
        # return array([[[ 0.01432707, -0.00312652],
        #                [ 0.01432707, -0.00312652]],
        #
        #               [[-0.00720589,  0.00887163],
        #                [-0.00720589,  0.00887163]],
        #
        #               [[ 0.00859588, -0.00636524],
        #                [ 0.00859588, -0.00636524]]]) shape:(3,2,2) = (src_sent_len,batch_size,embsize)
        # which means each index in indices matrix is changed into embsize vector

        return self.W[indices.flatten()].reshape(outshape)

    def apply_zero_pad(self, indices):
        # the embedding of first words of sentences in a batch are zeros
        # skip the last words (actually </S> ) of sentences
        '''
array([[[ 0.63626575,  0.51241083],
        [ 0.94934905,  0.56541414]],

       [[ 0.341131  ,  0.40076915],
        [ 0.17915587,  0.1175594 ]],

       [[ 0.61713483,  0.38632117],
        [ 0.29373151,  0.49611981]]]) 
----->
array([[[ 0.        ,  0.        ],
        [ 0.        ,  0.        ]],

       [[ 0.63626575,  0.51241083],
        [ 0.94934905,  0.56541414]],

       [[ 0.341131  ,  0.40076915],
        [ 0.17915587,  0.1175594 ]]])
        '''
        outshape = [indices.shape[i] for i
                    in range(indices.ndim)] + [self.embsize]

        emb = self.W[indices.flatten()].reshape(outshape)
        emb_shifted = T.zeros_like(emb)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb[:-1])
        return emb_shifted

    # def index(self, i): # the embedding (vector) of the word whose index is i in vocabulary
    #    return self.W[i]
    def index(self, y):  # the embedding (vector) of the word whose index is i in vocabulary
        emb = T.switch(y[:, None] < 0, T.alloc(0., 1, self.embsize),
                       self.W[y])

        '''
self.W:
array([[ 0.88694576,  0.29241001],
       [ 0.98971275,  0.57555197],
       [ 0.25140448,  0.34338728],
       [ 0.1344545 ,  0.44732701],
       [ 0.55294628,  0.38662105],
       [ 0.73885463,  0.73763555],
       [ 0.07260469,  0.63151152],
       [ 0.00645525,  0.83998468],
       [ 0.76262918,  0.32615177],
       [ 0.40990061,  0.41539679]])

y: array([2, 3, 8, 0, -1]), shape(5,)
y[:, None]: shape(5,1)
array([[ 2],
       [ 3],
       [-3],
       [ 0],
       [-1]])
emb: if less than 0, the row is 0 vector tensor
[array([[ 0.25140448,  0.34338728],
        [ 0.1344545 ,  0.44732701],
        [ 0.        ,  0.        ],
        [ 0.88694576,  0.29241001],
        [ 0.        ,  0.        ]])]
        '''
        # T.alloc(0., 1, 2): array([[ 0.,  0.]], dtype=float32), shape:array([1, 2])
        return emb


class BiGRU(object):

    # bidirectional gated recurrent unit: the purpose is encoding source sentence
        # n_in: the embedding size of input layer (source word embedding)
        # n_hids: the units number of source hiddens layer
    def __init__(self, n_in, n_hids, with_contex=False, prefix='BiGRU', **kwargs):
        kwargs['merge'] = False
        # forward direction gated recurrent unit (from left to right)
        self.encoder = GRU(n_in, n_hids, with_contex=with_contex,
                           prefix=_p(prefix, 'l2r'), **kwargs)
        # reverse direction gated recurrent unit (from right to left)
        self.rencoder = GRU(n_in, n_hids, with_contex=with_contex,
                            prefix=_p(prefix, 'r2l'), **kwargs)

        self.params = self.encoder.params + self.rencoder.params

    def apply(self, state_below, mask):
        # reverse by the first dimension of matrix, other dimension does not change
        # actually we reverse the sentence words order in each batch
        rstate_below = state_below[::-1]
        if mask is None:
            rmask = None
        else:
            rmask = mask[::-1]  # reverse corresponding mask as well
        # state_below:(src_sent_len,batch_size,embsize), mask:(src_sent_len,batch_size)
        loutput = self.encoder.apply(state_below, mask)
        # loutput: (src_sent_len, batch_size, n_hids)
        # rstate_below:(src_sent_len,batch_size,embsize),
        # rmask:(src_sent_len,batch_size), reverse word order in each sentence of
        # each batch
        routput = self.rencoder.apply(rstate_below, rmask)
        # routput: (src_sent_len, batch_size, n_hids)

        # [::-1] means reverse the tensor by the first dimension, actually reverse the hidden state tensor generated from right to left
        self.output = T.concatenate([loutput, routput[::-1]], axis=2)
        # self.output: shape (src_sent_len, batch_size, n_hids*2)
        return self.output


class LSTM(object):

    def __init__(self, n_hids=248, emb_size=124,
                 prefix='LSTM'):
        self.prefix = prefix
        self.n_hids = n_hids
        self.emb_size = emb_size
        self.params = []
        self._init_params()

    def _init_params(self):
        prefix = self.prefix
        w_size = (self.emb_size, self.n_hids)
        u_size = (self.n_hids, self.n_hids)
        self.W = param_init().param(w_size, init_type='mfunc',
                                    m=4, name=_p(prefix, 'W'))
        self.U = param_init().param(u_size, init_type='mfunc', m=4,
                                    name=_p(prefix, 'U'))
        self.b = param_init().param((self.n_hids * 4,),
                                    name=_p(prefix, 'b'))
        self.params += [self.W, self.U, self.b]

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(self, x_, m_, h_, c_):
        preact = T.dot(h_, self.U)
        preact += x_
        n_hids = self.n_hids
        _slice = self._slice

        i = T.nnet.sigmoid(_slice(preact, 0, n_hids))
        f = T.nnet.sigmoid(_slice(preact, 1, n_hids))
        o = T.nnet.sigmoid(_slice(preact, 2, n_hids))
        c = T.tanh(_slice(preact, 3, n_hids))
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    def forward(self, state_below, mask_below):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError

        below_x = T.dot(state_below, self.W) + self.b
        init_states = [T.alloc(numpy.float32(0.), batch_size, self.n_hids),
                       T.alloc(numpy.float32(0.), batch_size, self.n_hids)]
        rval, updates = theano.scan(self._step,
                                    sequences=[below_x, mask_below],
                                    outputs_info=init_states,
                                    n_steps=n_steps)
        return rval[0]


class DBLSTM(object):

    def __init__(self, emb_size, n_hids, n_layer=1, **kwargs):
        self.layers = []
        self.params = []
        n_layer = 1
        for i in range(n_layer):
            l_lstm = LSTM(n_hids, emb_size, prefix='DBLSTM_L_%d' % i)
            r_lstm = LSTM(n_hids, emb_size, prefix='DBLSTM_R_%d' % i)
            self.layers.append((l_lstm, r_lstm))
            self.params += l_lstm.params
            self.params += r_lstm.params

    def apply(self, state_below, mask_below):
        for l_lstm, r_lstm in self.layers:
            lstate_below = l_lstm.forward(state_below, mask_below)
            rstate_below = r_lstm.forward(state_below[::-1], mask_below[::-1])
            rstate_below = rstate_below[::-1]
        return T.concatenate([lstate_below, rstate_below], axis=2)
