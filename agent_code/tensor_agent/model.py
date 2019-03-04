import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e


from agent_code.tensor_agent.loss import mean_huber_loss


choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']

# channels: arena, self, others (3), bombs, explosions, coins -> c = 8 (see get_x)
c = 8

def create_conv_net(shape):
    inputs = Input(shape=shape)
    x = Conv2D(8, 3, activation='relu', padding="same")(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(16, 3, activation='relu', padding="same")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding="same")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding="same")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, 3, activation='relu', padding="same")(x)
    x = Dropout(0.3)(x)
    x = Conv2D(64, 3, activation='relu', padding="same")(x)
    x = Dropout(0.3)(x)
    outputs = Flatten()(x)

    return inputs, outputs

def create_value(x):
    v = Dense(64, activation='relu')(x)
    v = Dense(1, activation=None)(v)
    return v

def create_advantage(x, D):
    a = Dense(64, activation='relu')(x)
    a = Dense(D, activation=None)(a)
    return a

def create_model(shape, D):
        inputs, x = create_conv_net(shape=shape)

        v = create_value(x)
        a = create_advantage(x, D)

        outputs = Activation('relu')(v + a - tf.reduce_mean(a))

        model = Model(inputs=inputs, outputs=outputs)

        return model, inputs, outputs

class FullModel:
    def __init__(self, input_shape, D):
        #========================
        #  Define Model
        #========================

        self.online, _, _ = create_model(input_shape, D)
        self.target, t_in, t_out = create_model(input_shape, D)

        
        #========================
        #  Define Training Update
        #========================

        action_holder = Input(shape=(1,), dtype='int32')  # in j=0,...,D-1
        reward_holder = Input(shape=(1,))
        
        # applies a mask to the outputs so that only the prediction for the chosen action is considered
        responsible_weight = tf.gather(t_out, action_holder, axis=1)

        loss = mean_huber_loss(reward_holder, responsible_weight)

        optimizer = tf.train.AdamOptimizer(0.1)
        update = optimizer.minimize(loss)

        self.input_ph = t_in
        self.action_ph = action_holder
        self.reward_ph = reward_holder
        self.update_op = update


    def update(self, inputs, actions, rewards):
        sess = K.get_session()

        sess.run([self.update_op], feed_dict={
            self.input_ph: inputs,
            self.action_ph:actions,
            self.reward_ph:rewards
        })

    def update_online(self):
        self.online.set_weights(self.target.get_weights())
