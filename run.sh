python prepare_data.py
THEANO_FLAGS=mode=FAST_RUN,optimizer=fast_compile,device=gpu1,floatX=float32 python $1
