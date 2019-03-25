
#from tensorflow.keras import backend as K
import numpy as np

from tensorflow.keras.models import load_model


from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

from tensorflow.keras.initializers import RandomUniform, Constant
from tensorflow.keras import activations





# NoisyDense layer, in reference to
# https://github.com/cmusjtuliuyuan/RainBow/blob/master/model.py
# https://arxiv.org/pdf/1710.02298.pdf
# https://arxiv.org/pdf/1706.10295.pdf
class NoisyDense(Layer):
    def __init__(self, output_dim, activation=None, sigma_0=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.supports_masking = True

        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.sigma_0 = sigma_0

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

        sigma_init = Constant(self.sigma_0/(input_size ** 0.5))


        #========================
        # Weights
        #========================
        # According to formula
        # y = (b + Wx) + (b_sigma * b_epsilon + (W_sigma * W_epsilon)x)
        # => y = (b + b_sigma * b_epsilon) + (W + W_sigma * W_epsilon)x
        w_mu = self.add_weight(shape=(input_size, output_size), initializer=mu_init, name='w_mu')
        w_sigma = self.add_weight(shape=(input_size, output_size), initializer=sigma_init, name='w_sigma')

        self.w = w_mu + w_sigma * w_epsilon

        b_mu = self.add_weight(shape=(output_size,), initializer=mu_init, name='b_mu')
        b_sigma = self.add_weight(shape=(output_size,), initializer=sigma_init, name='b_sigma')

        self.b = b_mu + b_sigma * b_epsilon
        
        self.w_sigma = w_sigma
        self.b_sigma = b_sigma
        self.w_mu = w_mu
        self.b_mu = b_mu


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


class VAMerge(Layer):
    def __init__(self, **kwargs):
        super(VAMerge, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, training=None):
        v = inputs[0]
        a = inputs[1]
        return v + a - K.mean(a)


    def compute_output_shape(self, input_shape):
        return input_shape[1]











choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
D = len(choices)
cols, rows = 17, 17
bomb_timer = 4


class RelativeX3:
    shape = (cols * 2 - 1, rows * 2 - 1, 6)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, boxes, others, bombs, explosions, coins -> c = 6
        X = np.zeros((cols, rows, 6))
        
        
        X[:,:,0] = arena == -1

        X[:,:,1] = arena == 1
        
        for i in range(len(others)):
            X[others[i][0], others[i][1], 2] = 1 + (others[i][3] > 0)
        
        for i in range(len(bombs)):
            X[bombs[i][0], bombs[i][1], 3] = 1 - bombs[i][2] / (bomb_timer+1)
        
        X[:,:,4] = np.maximum(explosions - 1, 0)
        
        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 5] = 1
        
        x, y, _, _, _ = self

        centered_X = np.zeros((cols * 2 - 1, rows * 2 - 1, 6))
        centered_X[cols-1-x:cols*2-1-x, rows-1-y:rows*2-1-y] = X
        return centered_X


def get_valid_actions(game_state):
    valid = np.ones((D))
    x, y, _, b, _ = game_state['self']
    arena = game_state['arena']
    explosions = game_state['explosions']
    if not tile_is_free(x, y-1, game_state) or explosions[x, y-1] > 1:
        valid[2] = 0 # UP invalid
    if not tile_is_free(x, y+1, game_state) or explosions[x, y+1] > 1:
        valid[3] = 0 # DOWN invalid
    if not tile_is_free(x-1, y, game_state) or explosions[x-1, y] > 1:
        valid[1] = 0 # LEFT invalid
    if not tile_is_free(x+1, y, game_state) or explosions[x+1, y] > 1:
        valid[0] = 0 # RIGHT invalid
    if b <= 0:
        valid[4] = 0

    if np.any(valid[0:4]) and explosions[x, y] > 1:
        valid[5] = 0

    return valid

def tile_is_free(x, y, game_state):
    is_free = game_state['arena'][x,y] == 0
    if is_free:
        for obstacle in game_state['bombs']:
            o_x, o_y, _ = obstacle
            is_free = is_free and (o_x != x or o_y != y)
    return is_free

def setup(self):
    #K.clear_session()
    
    
    #========================
    #  Define Model
    #========================
    
    model = load_model('bomberchamp.h5', custom_objects={'VAMerge': VAMerge, 'NoisyDense': NoisyDense})

    self.model = model

def act(self):
    X = RelativeX3.get(self.game_state)

    valid_actions = get_valid_actions(self.game_state)

    pred = self.model.predict(np.array([X]))[0]
    pred = valid_actions * (pred - np.min(pred) + 1)

    action_choice = np.argmax(pred)

    self.next_action = choices[action_choice]

def reward_update(self):
    pass

def end_of_episode(self):
    pass
