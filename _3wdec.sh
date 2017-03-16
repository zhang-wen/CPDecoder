#THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=5000 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python wtrans.py 13 263653 search_model_ch2en/params_e13_upd263653.npz $1 $2 $3
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 10 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 20 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 50 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 100 1
#THEANO_FLAGS=device=cpu,floatX=float32 python __trans__.py 13 263653 search_model_ch2en/params_e13_upd263653.npz 500 1
#--model-name search_model_ch2en/params_e13_upd263653.npz \
		#--model-name search_model_ch2en/params_e1_upd20112.npz \
		#--model-name ./wmodels/params_e13_upd260000.npz \
		# Him1   Hi     AiKL
		#--model-name ./params_e11_upd223091.npz \
THEANO_FLAGS=mode=FAST_RUN,device=gpu3,floatX=float32,lib.cnmem=2000 python wtrans.py \
		--epoch 11 \
		--batch 223091 \
		--search-mode $1 \
		--model-name ./params_e11_upd223091_38.80.npz \
		--beam-size $2 \
		--use-norm 1 \
		--use-batch 0 \
		--use-score 0 \
		--use-valid 1 \
		--valid-set $3 \
		--use-mv 0 \
		--ifwatch-adist 0 \
		--merge-way 'Him1' \
        --ifapprox-dist 0 \
        --ifapprox-att 0 \
        --ifadd-lmscore 0 \
		--m-threshold 10000.0 \
		--n-process 10 \
		--ngram 2 \
		--length-norm 0. \
		--cover-penalty 0. \
		--workspace $4 \
		#--lm-path '/home/wen/3.corpus/train.en.a2b.low.jiutok.pad.o5.binary'

