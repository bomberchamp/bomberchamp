from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras import activations

# NoisyDense layer, in reference to
# https://github.com/cmusjtuliuyuan/RainBow/blob/master/model.py
# https://arxiv.org/pdf/1710.02298.pdf
# https://arxiv.org/pdf/1706.10295.pdf
class NoisyDense(Layer):
    def __init__(self, output_dim, activation=None, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.supports_masking = True

        self.output_dim = output_dim
        self.activation = activations.get(activation)

    def build(self, input_shape):
        input_size = int(input_shape[-1])
        output_size = self.output_dim

        #========================
        # Calculate epsilon
        #========================
        # Fortunado et al. 2017, eq. 10+11
        # https://arxiv.org/pdf/1706.10295.pdf
        def f(x):
            return K.sign(x) * K.pow(K.abs(x), 0.5)
        
        p = K.random_normal(shape=(input_size, 1))
        q = K.random_normal(shape=(1, output_size))
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = K.squeeze(f_q, 0)
        


        #========================
        # Initializer
        #========================
        # Fortunado et al. 2017, chapter 3.2
        # https://arxiv.org/pdf/1706.10295.pdf
        #
        # sigma_0 is a hyperparameter
        low = -1*1/(input_size ** 0.5)
        high = 1*1/(input_size ** 0.5)
        mu_init = RandomUniform(minval=low,maxval=high)

        sigma_0 = 0.5
        sigma_init = Constant(sigma_0/(input_size ** 0.5))


        #========================
        # Weights
        #========================
        # According to formula
        # y = (b + Wx) + (b_sigma * b_epsilon + (W_sigma * W_epsilon)x)
        # => y = (b + b_sigma * b_epsilon) + (W + W_sigma * W_epsilon)x
        w_mu = self.add_weight(shape=(input_size, output_size), initializer=mu_init)
        w_sigma = self.add_weight(shape=(input_size, output_size), initializer=sigma_init)

        self.w = w_mu + w_sigma * w_epsilon

        b_mu = self.add_weight(shape=(output_size,), initializer=mu_init)
        b_sigma = self.add_weight(shape=(output_size,), initializer=sigma_init)

        self.b = b_mu + b_sigma * b_epsilon
        


    def call(self, inputs, training=None):
        # since the weights are already combined, this is the same as a standard Dense layer:
        # y = b + Wx
        output = K.dot(inputs, self.w) + self.b
        return self.activation(output)


    def compute_output_shape(self, input_shape):
        # Same as Keras Dense layer
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'activation': self.activation
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


