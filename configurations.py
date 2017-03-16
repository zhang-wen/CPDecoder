def get_config_cs2en():
    config = {}

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['seq_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['enc_nhids'] = 1024
    config['dec_nhids'] = 1024

    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 512
    config['dec_embed'] = 512

    # dimension of the output layer
    config['n_out'] = 512

    # Where to save model, this corresponds to 'prefix' in groundhog
    config['models_dir'] = 'wmodels'
    # test output dir
    config['test_dir'] = ''

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 20

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Regularization related --------------------------------------------------

    # layer normalization
    config['ln'] = False

    # Dropout ratio, applied only after readout maxout
    config['dropout'] = 0.

    # super-parameter for Self-Normalized, apply for not needing to use softmax when decoding
    config['alpha'] = 0.005

    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    config['datadir'] = './data/'

    config['prepare_file'] = './prepare_data.py'
    config['preprocess_file'] = './dict_preprocess.py'

    # Source and target vocabularies
    config['src_vocab'] = config['datadir'] + 'vocab.zh-en.zh.pkl'
    config['trg_vocab'] = config['datadir'] + 'vocab.zh-en.en.pkl'

    # Source and target datasets
    config['src_data'] = config['datadir'] + 'train.zh'
    config['trg_data'] = config['datadir'] + 'train.en'
    config['dict_data'] = config['datadir'] + 'train.sent.dict'

    # valids.dict
    config['valid_sent_dict'] = config['datadir'] + 'valids.dict'

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = 30000
    config['trg_vocab_size'] = 30000

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # decoding parameters -------------------------------------------
    # 0: MLE, 1: Original, 2:Naive, 3:cube pruning
    config['search_mode'] = 2
    # Beam-size
    config['beam_size'] = 10

    config['use_batch'] = 0
    config['use_score'] = 1  # because we add self-normalized in training
    # Normalize cost according to sequence length after beam-search
    config['use_norm'] = 1
    # whether manipulate vocabulary or not
    config['use_mv'] = 0
    config['watch_adist'] = False
    # Him1, Hi, AiKL, LM
    config['merge_way'] = 'Him1'
    config['avg_att'] = False
    config['m_threshold'] = 100000.
    config['ngram'] = 4
    config['length_norm'] = 0.
    config['cover_penalty'] = 0.
    config['lm_path'] = '/scratch2/wzhang/1.research/4.mt/clause-baesd-mt/lm/train.en.a2b.low.jiutok.o5.binary'

    # Print validation output to file
    config['output_val_set'] = True

    # Validation and test file dir
    config['val_tst_dir'] = '/home/wen/3.corpus/allnist_stanfordseg_jiujiu/'
    # for testing 50 sentence of nist02
    #config['val_tst_dir'] = '/scratch2/wzhang/3.corpus/2.mt/nist-all/test/'

    # Validation prefix
    config['val_prefix'] = 'nist02'

    # Model prefix
    config['model_prefix'] = config['models_dir'] + '/params'

    # Validation set source file
    config['val_set'] = config['val_tst_dir'] + config['val_prefix'] + '.src'

    # Valid output directory
    config['val_out_dir'] = 'wvalids'
    # Test output directory
    config['tst_out_dir'] = ''

    # Maximum number of updates
    config['max_epoch'] = 20

    # Reload model from files if exist
    config['reload'] = False
    config['one_model'] = './params_e23_upd466463.npz'
    config['save_one_model'] = False
    config['epoch_eval'] = False

    # Save model after this many updates
    config['save_freq'] = 10000
    #config['save_freq'] = 100

    # about 22500 batches for one epoch
    # Show samples from model after this many updates
    config['sampling_freq'] = 10000
    #config['sampling_freq'] = 50

    # Start bleu validation after this many updates
    #config['val_burn_in'] = 10000
    config['val_burn_in'] = 2

    # Show details
    config['display_freq'] = 1000
    #config['display_freq'] = 1

    # Show this many samples at each sampling, need less than batch size
    config['hook_samples'] = 5

    # Validate bleu after this many updates
    config['bleu_val_freq'] = 10000
    #config['bleu_val_freq'] = 100

    # whether use fixed sampling or randomly sampling
    config['if_fixed_sampling'] = True

    # Start fix sampling after this many batches
    config['k_batch_start_sample'] = 1000

    # the v^T in Haitao's paper, the top k target words merged into target vocabulary
    config['topk_trg_vocab'] = 50
    config['topk_trg_pkl'] = config['datadir'] + \
        str(config['topk_trg_vocab']) + 'vocab.zh-en.en.pkl'
    config['trg_cands_dict'] = config['datadir'] + 'dict2dict_cands.dict'
    config['trg_cands_pkl'] = config['datadir'] + 'dict2dict_cands.pkl'
    config['lex_f2e'] = '/scratch2/wzhang/1.research/7.extract-phrase/1.8m-moses-nosmooth/moses-train/model/lex.f2e'
    config['phrase_table'] = '/scratch2/wzhang/1.research/7.extract-phrase/1.8m-moses-nosmooth/moses-train/model/phrase-table.gz'

    return config
