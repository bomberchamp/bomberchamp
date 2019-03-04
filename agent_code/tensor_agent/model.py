import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K


from agent_code.tensor_agent.loss import mean_huber_loss
from agent_code.tensor_agent.layers import NoisyDense, VAMerge


def create_conv_net(shape):
    # Convolutional part of the network
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

def create_stream(x, D):
    s = NoisyDense(64, activation='relu')(x)
    s = NoisyDense(D, activation=None)(s)
    return s

def create_model(shape, D):
    # Create the convolutional network
    inputs, x = create_conv_net(shape=shape)

    # Dueling networks:
    # - Split the model into value stream and advantage stream
    v = create_stream(x, 1)
    a = create_stream(x, D)

    # - Merge streams
    outputs = VAMerge()([v, a])
    outputs = Activation('relu')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    return model, inputs, outputs

class FullModel:
    def __init__(self, input_shape, D):
        #========================
        #  Define Model
        #========================

        # Define online and target models for Double Q-learning
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
