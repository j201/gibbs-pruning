import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tensorflow.python.ops import nn

def tf_sample_gibbs(A, b, beta, N):
    """Naive sampling from p(x) = 1/Z*exp(-beta*(x^T*A*x + b*x)"""
    xs = K.constant(list(itertools.product([-1,1], repeat=N))) # 2^N x N tensor
    quad = -beta * K.sum(tf.tensordot(xs, A, axes=[[1],[0]]) * xs[:,:,None,None], axis=1)
    quad = quad - K.max(quad, axis=[0])[None,:,:] # Put the highest quad logits around 0 to ensure precision when we add biases
    logits = quad - beta*tf.tensordot(xs, b, axes=[[1],[0]])
    logits = logits - K.max(logits, axis=[0]) # Same, tensorflow doesn't seem to work well with high logits
    rows = tf.random.categorical(K.transpose(K.reshape(logits, (2**N,-1))), 1)[:,0]
    slices = tf.gather(xs, rows, axis=0)
    return K.reshape(K.transpose(slices), K.shape(b))

class GibbsPrunedConv2D(layers.Conv2D):
    # TODO: document
    # Mention that efficiency gains aren't actually implemented

    def __init__(self, filters, kernel_size, p=0.5, hamiltonian='unstructured',
            c=1.0, test_pruning_mode='gibbs', mcmc_steps=50, **kwargs):
        self.p = p
        self.hamiltonian = hamiltonian
        self.c = c
        self.test_pruning_mode = test_pruning_mode
        self.mcmc_steps = mcmc_steps
        self.beta = tf.Variable(1.0, trainable=False, name='beta') # This will be updated before training by the annealer
        super().__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        self.n_channels = int(input_shape[channel_axis])
        self.mask = K.zeros_like(self.kernel)

    def call(self, inputs):
        # This code appears in the Keras Conv layer, but is only compatible
        # with TF2. I'm not sure what situations it addresses, but it doesn't
        # seem necessary for the example code in this repo
        # # Check if the input_shape in call() is different from that in build().
        # # If they are different, recreate the _convolution_op to avoid the stateful
        # # behavior.
        # call_input_shape = inputs.get_shape()
        # recreate_conv_op = (
        #         call_input_shape[1:] != self._build_conv_op_input_shape[1:])
        # if recreate_conv_op:
        #     self._convolution_op = nn_ops.Convolution(
        #             call_input_shape,
        #             filter_shape=self.kernel.shape,
        #             dilation_rate=self.dilation_rate,
        #             strides=self.strides,
        #             padding=self._padding_op,
        #             data_format=self._conv_op_data_format)

        mask = K.in_train_phase(lambda: self.train_mask(), lambda: self.test_mask())
        self.add_metric(1-K.mean(mask), name='gp_mask_p', aggregation='mean')
        self.add_metric(self.beta, name='beta', aggregation='mean')
        outputs = self._convolution_op(inputs, self.kernel * mask)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def test_mask(self):
        W2 = self.kernel * self.kernel
        if self.test_pruning_mode == 'gibbs':
            return self.train_mask()
        elif self.test_pruning_mode == 'kernel':
            kernel_sums = tf.reduce_sum(W2, axis=[0,1])
            Qp = tfp.stats.percentile(kernel_sums, self.p*100, interpolation='linear')
            return K.cast(kernel_sums >= Qp, 'float32')[None,None,:,:]
        elif self.test_pruning_mode == 'filter':
            filter_sums = tf.reduce_sum(W2, axis=[0,1,2])
            Qp = tfp.stats.percentile(filter_sums, self.p*100, interpolation='linear')
            return K.cast(filter_sums >= Qp, 'float32')[None,None,None,:]
        else:
            raise ValueError("test_pruning_mode must be one of 'gibbs', 'kernel', or 'filter'")

    def train_mask(self):
        W2 = self.kernel * self.kernel
        n_filter_weights = np.product(self.kernel_size)
        if self.hamiltonian == 'unstructured':
            Qp = tfp.stats.percentile(K.flatten(W2), self.p*100, interpolation='linear')
            P0 = 1/(1+K.exp(self.beta*(W2-Qp)))
            R = K.random_uniform(K.shape(P0))
            return K.cast(R > P0, 'float32')
        elif self.hamiltonian == 'kernel':
            # Prune kernels by finding A and B for hamiltonian H(x) = x^TAx +
            # b^Tx, and sampling directly for each kernel
            flat_W2 = K.reshape(W2, (n_filter_weights, self.n_channels, self.filters))
            Qp = tfp.stats.percentile(K.sum(flat_W2,axis=0)/n_filter_weights, self.p*100, interpolation='linear')
            b = Qp - flat_W2
            A = -self.c * K.constant(np.ones((n_filter_weights, n_filter_weights, self.n_channels, self.filters)))
            A_mask = np.ones((n_filter_weights,n_filter_weights))
            np.fill_diagonal(A_mask, False)
            A = A * A_mask[:,:,None,None]
            M = K.reshape(tf_sample_gibbs(A, b, self.beta, n_filter_weights), K.shape(W2))
            return (M+1)/2
        elif self.hamiltonian == 'filter':
            # Prune filters with chromatic gibbs sampling
            flat_W2 = K.reshape(W2, (n_filter_weights, self.n_channels, self.filters))
            Qp = tfp.stats.percentile(tf.reduce_sum(flat_W2,axis=[0,1])/n_filter_weights/self.n_channels, self.p*100, interpolation='linear')
            b = Qp - flat_W2
            A = -self.c * K.constant(np.ones((n_filter_weights, n_filter_weights, self.n_channels, self.filters)))
            A_mask = np.ones((n_filter_weights,n_filter_weights))
            np.fill_diagonal(A_mask, False)
            A = A * A_mask[:,:,None,None]

            filt_avgs = tf.reduce_sum(flat_W2,axis=[0,1])/n_filter_weights/self.n_channels
            x_cvg = K.cast(filt_avgs > Qp, 'float32')
            colour_b = b - self.c * (self.n_channels//2) * n_filter_weights * (x_cvg*2-1)[None,None,:]

            split = self.n_channels//2
            colour_b = colour_b[:,0:split,:]
            for i in range((self.mcmc_steps)+1//2):
                M0 = tf_sample_gibbs(A[:,:,0:split,:], colour_b, self.beta, n_filter_weights)
                filter_sums = tf.reduce_sum(M0, axis=[0,1])
                colour_b = b[:,split:,:] - self.c*filter_sums[None,None,:]
                M1 = tf_sample_gibbs(A[:,:,split:,:], colour_b, self.beta, n_filter_weights)
                filter_sums = tf.reduce_sum(M1, axis=[0,1])
                colour_b = b[:,0:split,:] - self.c*filter_sums[None,None,:]
            M = K.reshape(K.concatenate((M0,M1), axis=1), K.shape(W2))
            return (M+1)/2

    def get_config(self):
        config = {
            'beta_init': self.beta_init,
            'p': self.p,
            'hamiltonian': self.hamiltonian,
            'c': self.c,
            'test_pruning_mode': self.test_pruning_mode,
            'mcmc_steps': self.mcmc_steps,
        }
        base_config = super().get_config()
        return {**config, **base_config}

    def set_beta(self, beta):
        K.set_value(self.beta, beta)

class GibbsPruningAnnealer(keras.callbacks.Callback):
    def __init__(self, beta_schedule, verbose=0):
        super().__init__()
        self.beta_schedule = beta_schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        beta = self.beta_schedule[epoch] if epoch < len(self.beta_schedule) else self.beta_schedule[-1]
        count = 0
        for layer in self.model.layers:
            if isinstance(layer, GibbsPrunedConv2D):
                count += 1
                layer.set_beta(beta)
        if self.verbose > 0:
            print(f'GibbsPruningAnnealer: set beta to {beta} in {count} layers')
