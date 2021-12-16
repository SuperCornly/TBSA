
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer
import tensorflow as tf

class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                layer.build(K.int_shape(args[0]))
            else:
                layer.build(K.int_shape(kwargs['inputs']))
            self._trainable_weights.extend(layer._trainable_weights)
            self._non_trainable_weights.extend(layer._non_trainable_weights)
        return layer.call(*args, **kwargs)

class AttentionPooling1D(OurLayer):
    def __init__(self, h_dim=None, **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
    def build(self, input_shape):
        super(AttentionPooling1D, self).build(input_shape)
        if self.h_dim is None:
            self.h_dim = input_shape[-1]
        self.k_dense = Dense(
            self.h_dim,
            use_bias=False,
            activation='tanh'
        )
        self.o_dense = Dense(1, use_bias=False)
    def call(self, inputs):
        xo = inputs
        x = xo
        x = self.reuse(self.k_dense, x)
        x = self.reuse(self.o_dense, x)
        #x = x - (1 - mask) * 1e12
        x = K.softmax(x, 1)
        return K.sum(x * xo, 1)
    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


class DilatedGatedConv1D(OurLayer):
    def __init__(self,
                 o_dim=None,
                 k_size=3,
                 rate=1,
                 skip_connect=True,
                 drop_gate=None,
                 **kwargs):
        super(DilatedGatedConv1D, self).__init__(**kwargs)
        self.o_dim = o_dim
        self.k_size = k_size
        self.rate = rate
        self.skip_connect = skip_connect
        self.drop_gate = drop_gate
    def build(self, input_shape):
        super(DilatedGatedConv1D, self).build(input_shape)
        if self.o_dim is None:
            self.o_dim = input_shape[0][-1]
        self.conv1d = Conv1D(
            self.o_dim * 2,
            self.k_size,
            dilation_rate=self.rate,
            padding='same'
        )
        if self.skip_connect and self.o_dim != input_shape[-1]:
            self.conv1d_1x1 = Conv1D(self.o_dim, 1)
    def call(self, inputs):
        xo= inputs
        x = xo 
        x = self.reuse(self.conv1d, x)
        x, g = x[..., :self.o_dim], x[..., self.o_dim:]
        if self.drop_gate is not None:
            g = K.in_train_phase(K.dropout(g, self.drop_gate), g)
        g = K.sigmoid(g)
        if self.skip_connect:
            if self.o_dim != K.int_shape(xo)[-1]:
                xo = self.reuse(self.conv1d_1x1, xo)
            return (xo * (1 - g) + x * g) 
        else:
            return x * g
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.o_dim,)


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    return scale * x


#define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)


#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def cumsoftmax(x, mode='l2r'):
    """先softmax，然后cumsum，
    cumsum区分从左到右、从右到左两种模式
    """
    axis = K.ndim(x) - 1
    if mode == 'l2r':
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x
    elif mode == 'r2l':
        x = x[..., ::-1]
        x = K.softmax(x, axis=axis)
        x = K.cumsum(x, axis=axis)
        return x[..., ::-1]
    else:
        return x


class ONLSTM(Layer):
    """实现有序LSTM，来自论文
    Ordered Neurons: Integrating Tree Structures into Recurrent Neural Networks
    """
    def __init__(self,
                 units,
                 levels,
                 return_sequences=False,
                 dropconnect=None,
                 bench=64,
                 **kwargs):
        assert units % levels == 0
        self.bench=bench
        self.units = units
        self.levels = levels
        self.chunk_size = units // levels
        self.return_sequences = return_sequences
        self.dropconnect = dropconnect
        super(ONLSTM, self).__init__(**kwargs)
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4 + self.levels * 2),
            name='kernel',
            initializer='glorot_uniform')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4 + self.levels * 2),
            name='recurrent_kernel',
            initializer='orthogonal')
        self.bias = self.add_weight(
            shape=(self.units * 4 + self.levels * 2,),
            name='bias',
            initializer='zeros')
        self.built = True
        if self.dropconnect:
            self._kernel = K.dropout(self.kernel, self.dropconnect)
            self._kernel = K.in_train_phase(self._kernel, self.kernel)
            self._recurrent_kernel = K.dropout(self.recurrent_kernel, self.dropconnect)
            self._recurrent_kernel = K.in_train_phase(self._recurrent_kernel, self.recurrent_kernel)
        else:
            self._kernel = self.kernel
            self._recurrent_kernel = self.recurrent_kernel
        self.initial_states = [
            K.zeros((self.bench, self.units)),
            K.zeros((self.bench, self.units))
        ] # 定义初始态(全零)
    def one_step(self, inputs, states):
        x_in, (c_last, h_last) = inputs, states
        x_out = K.dot(x_in, self._kernel) + K.dot(h_last, self._recurrent_kernel)
        x_out = K.bias_add(x_out, self.bias)
        f_master_gate = cumsoftmax(x_out[:, :self.levels], 'l2r')
        f_master_gate = K.expand_dims(f_master_gate, 2)
        i_master_gate = cumsoftmax(x_out[:, self.levels: self.levels * 2], 'r2l')
        i_master_gate = K.expand_dims(i_master_gate, 2)
        x_out = x_out[:, self.levels * 2:]
        x_out = K.reshape(x_out, (-1, self.levels * 4, self.chunk_size))
        f_gate = K.sigmoid(x_out[:, :self.levels])
        i_gate = K.sigmoid(x_out[:, self.levels: self.levels * 2])
        o_gate = K.sigmoid(x_out[:, self.levels * 2: self.levels * 3])
        c_in = K.tanh(x_out[:, self.levels * 3:])
        c_last = K.reshape(c_last, (-1, self.levels, self.chunk_size))
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + \
                (f_master_gate - overlap) * c_last + \
                (i_master_gate - overlap) * c_in
        h_out = o_gate * K.tanh(c_out)
        c_out = K.reshape(c_out, (-1, self.units))
        h_out = K.reshape(h_out, (-1, self.units))
        out = K.concatenate([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, [c_out, h_out]
    def call(self, inputs):

        outputs = K.rnn(self.one_step, inputs, self.initial_states)
        self.distance = 1 - K.mean(outputs[1][..., self.units: self.units + self.levels], -1)
        self.distance_in = K.mean(outputs[1][..., self.units + self.levels:], -1)
        if self.return_sequences:
            return outputs[1][..., :self.units]
        else:
            return outputs[0][..., :self.units]
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return (input_shape[0], input_shape[1], self.units)
        else:
            return (input_shape[0], self.units)


