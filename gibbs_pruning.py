import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import tensorflow_probability as tfp

class GibbsPrunedConv2D(layers.Conv2D):
    # TODO: document
    # Mention that efficiency gains aren't actually implemented

    def __init__(self, filters, kernel_size, p=0.5, hamiltonian='unstructured',
            c=1.0, train_pruning_mode='gibbs', mcmc_steps=20, **kwargs):
        self.p = p
        self.hamiltonian = hamiltonian
        self.c = c
        self.train_pruning_mode = train_pruning_mode
        self.mcmc_steps = mcmc_steps
        self.beta = tf.Variable(1.0, trainable=False, name='beta') # This will be updated before training by the annealer
        super().__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        self.mask = K.zeros_like(self.kernel)

    def call(self, inputs):
        # Check if the input_shape in call() is different from that in build().
        # If they are different, recreate the _convolution_op to avoid the stateful
        # behavior.
        call_input_shape = inputs.get_shape()
        recreate_conv_op = (
                call_input_shape[1:] != self._build_conv_op_input_shape[1:])

        if recreate_conv_op:
            self._convolution_op = nn_ops.Convolution(
                    call_input_shape,
                    filter_shape=self.kernel.shape,
                    dilation_rate=self.dilation_rate,
                    strides=self.strides,
                    padding=self._padding_op,
                    data_format=self._conv_op_data_format)

        mask = K.in_train_phase(lambda: self.train_mask(), lambda: self.test_mask())
        self.add_metric(1-K.mean(mask), name='gp_mask_p', aggregation='mean')
        outputs = self._convolution_op(inputs, self.kernel * mask)

        if self.use_bias:
            if self.data_format == 'channels_first':
                outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def train_mask(self):
        W2 = self.kernel * self.kernel
        if self.train_pruning_mode == 'gibbs':
            return self.test_mask()
        elif self.train_pruning_mode == 'kernel':
            kernel_sums = tf.reduce_sum(W2, axis=[0,1])
            Qp = tfp.stats.percentile(kernel_sums, self.p*100, interpolation='linear')
            return K.cast(kernel_sums >= Qp, 'float32')[None,None,:,:]
        elif self.train_pruning_mode == 'filter':
            filter_sums = tf.reduce_sum(W2, axis=[0,1,2])
            Qp = tfp.stats.percentile(filter_sums, self.p*100, interpolation='linear')
            return K.cast(filter_sums >= Qp, 'float32')[None,None,None,:]
        else:
            raise ValueError("train_pruning_mode must be one of 'gibbs', 'kernel', or 'filter'")

    def test_mask(self):
        W2 = self.kernel * self.kernel
        Qp = tfp.stats.percentile(K.flatten(W2), self.p*100)
        n_filter_weights = np.product(self.kernel_size)
        if self.hamiltonian == 'unstructured':
            P0 = 1/(1+K.exp(self.beta*(W2-Qp)))
            R = K.random_uniform(K.shape(P0))
            return K.cast(R > P0, 'float32')

    def get_config(self):
        config = {
            'beta_init': self.beta_init,
            'p': self.p,
            'hamiltonian': self.hamiltonian,
            'c': self.c,
            'train_pruning_mode': self.train_pruning_mode,
            'mcmc_steps': self.mcmc_steps,
        }
        base_config = super().get_config()
        return {**config, **base_config}

    def set_beta(self, beta):
        self.beta.assign(beta)

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
