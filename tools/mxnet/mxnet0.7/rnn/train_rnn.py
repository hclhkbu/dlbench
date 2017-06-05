"""A simple demo of new RNN cell with PTB language model."""
import os
import argparse

import numpy as np
import mxnet as mx

from bucket_io import MyBucketSentenceIter, BucketSentenceIter, default_build_vocab


#os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "1"

#data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))


def Perplexity(label, pred):
    # collapse the time, batch dimension
    label = label.reshape((-1,))
    pred = pred.reshape((-1, pred.shape[-1]))

    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train rnn lstm with ptb')
    parser.add_argument('--data-dir', type=str, help='the input data directory')
    parser.add_argument('--gpus', type=str,
                                help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--sequence-lens', type=str, default="32",
                                help='the sequence lengths, e.g "8,16,32,64,128"')
    parser.add_argument('--batch-size', type=int, default=128,
                                help='the batch size')
    parser.add_argument('--num-hidden', type=int, default=256,
                                help='size of the state for each lstm layer')
    parser.add_argument('--num-embed', type=int, default=256,
                                help='dim of embedding')
    parser.add_argument('--num-lstm-layer', type=int, default=2,
                                help='the numebr of lstm layers')
    parser.add_argument('--lr', type=float, default=0.01,
                                help='learning rate')
    parser.add_argument('--model-prefix', type=str,
                                help='the prefix of the model to load')
    parser.add_argument('--num-examples', type=str,
                                help='Flag for consistancy, no use in rnn')
    parser.add_argument('--save-model-prefix', type=str,
                                help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=20,
                                help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int,
                                help='load the model on an epoch using the model-prefix')
    parser.add_argument('--kv-store', type=str, default='local',
                                help='the kvstore type')
    args = parser.parse_args()

    data_dir = os.environ['HOME'] + "/data/mxnet/ptb/" if args.data_dir is None else args.data_dir
    batch_size = args.batch_size  
    #buckets = [64] #[10, 20, 30, 40, 50, 60]
    buckets = [int(i) for i in args.sequence_lens.split(',')] #[8,16,32,64,128] 
    num_hidden = args.num_hidden
    num_embed = args.num_embed
    num_lstm_layer = args.num_lstm_layer 

    num_epoch = args.num_epochs
    learning_rate = args.lr
    momentum = 0.0

    contexts = mx.context.cpu() if args.gpus is None else [mx.context.gpu(int(i)) for i in args.gpus.split(',')]
    vocab = default_build_vocab(os.path.join(data_dir, 'ptb.train.txt'))
    print("Size of ptb.train.txt vocab: " + str(len(vocab)))

    init_h = [('LSTM_state', (num_lstm_layer, batch_size, num_hidden))]
    init_c = [('LSTM_state_cell', (num_lstm_layer, batch_size, num_hidden))]
    init_states = init_c + init_h

    data_train = MyBucketSentenceIter(os.path.join(data_dir, 'ptb.train.txt'),
                                    vocab, buckets, batch_size, init_states,
                                    time_major=True)
    data_val = MyBucketSentenceIter(os.path.join(data_dir, 'ptb.valid.txt'),
                                  vocab, buckets, batch_size, init_states,
                                  time_major=True)
    sample_size = 0
    for x in data_train.data:
        sample_size += len(x)
    print("len of data train===================== " + str(sample_size))
    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('softmax_label')
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab),
                                 output_dim=num_embed, name='embed')

        # TODO(tofix)
        # currently all the LSTM parameters are concatenated as
        # a huge vector, and named '<name>_parameters'. By default
        # mxnet initializer does not know how to initilize this
        # guy because its name does not ends with _weight or _bias
        # or anything familiar. Here we just use a temp workaround
        # to create a variable and name it as LSTM_bias to get
        # this demo running. Note by default bias is initialized
        # as zeros, so this is not a good scheme. But calling it
        # LSTM_weight is not good, as this is 1D vector, while
        # the initialization scheme of a weight parameter needs
        # at least two dimensions.
        rnn_params = mx.sym.Variable('LSTM_bias')

        # RNN cell takes input of shape (time, batch, feature)
        rnn = mx.sym.RNN(data=embed, state_size=num_hidden,
                         num_layers=num_lstm_layer, mode='lstm',
                         name='LSTM', 
                         # The following params can be omitted
                         # provided we do not need to apply the
                         # workarounds mentioned above
                         parameters=rnn_params)

        # the RNN cell output is of shape (time, batch, dim)
        # if we need the states and cell states in the last time
        # step (e.g. when building encoder-decoder models), we
        # can set state_outputs=True, and the RNN cell will have
        # extra outputs: rnn['LSTM_output'], rnn['LSTM_state']
        # and for LSTM, also rnn['LSTM_state_cell']

        # now we collapse the time and batch dimension to do the
        # final linear logistic regression prediction
        hidden = mx.sym.Reshape(data=rnn, shape=(-1, num_hidden))

        pred = mx.sym.FullyConnected(data=hidden, num_hidden=len(vocab),
                                     name='pred')

        # reshape to be of compatible shape as labels
        pred_tm = mx.sym.Reshape(data=pred, shape=(seq_len, -1, len(vocab)))

        sm = mx.sym.SoftmaxOutput(data=pred_tm, label=label, preserve_shape=True,
                                  name='softmax')

        data_names = ['data', 'LSTM_state', 'LSTM_state_cell']
        label_names = ['softmax_label']

        return (sm, data_names, label_names)

    if len(buckets) == 1:
        mod = mx.mod.Module(*sym_gen(buckets[0]), context=contexts)
    else:
        mod = mx.mod.BucketingModule(sym_gen, 
                                     default_bucket_key=data_train.default_bucket_key,
                                     context=contexts)

    print(args)
    print("Start training...")
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod.fit(data_train, eval_data=data_val, num_epoch=num_epoch,
            eval_metric=mx.metric.np(Perplexity),
            batch_end_callback=mx.callback.Speedometer(batch_size, int((sample_size-1)/batch_size)),
            #initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            initializer=mx.init.Uniform(scale=0.1),
            optimizer='sgd',
	    optimizer_params={'learning_rate': learning_rate, 'momentum': momentum, 'wd': 0.00001, 'clip_gradient': 5.0})
