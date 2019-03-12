import numpy as np
import tensorflow as tf

import time
from copy import copy

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from agent_code.tensor_agent.hyperparameters import hp

from agent_code.tensor_agent.loss import weighted_huber_loss, huber_loss
from agent_code.tensor_agent.layers import NoisyDense, VAMerge


def create_conv_net(shape):
    # Convolutional part of the network
    inputs = Input(shape=shape)
    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(256, activation='relu')(x)

    return inputs, outputs

def create_stream(x, D):
    
    s = NoisyDense(512, activation='relu')(x)
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
    #outputs = Activation('relu')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model, inputs, outputs

class FullModel:
    def __init__(self, input_shape, D):
        self.input_shape = input_shape
        self.D = D

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
        weight_holder = Input(shape=(1,))
        
        # applies a mask to the outputs so that only the prediction for the chosen action is considered
        responsible_weight = tf.batch_gather(t_out, action_holder)
        
        loss = weighted_huber_loss(reward_holder, responsible_weight, weight_holder)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('reward', tf.reduce_mean(reward_holder))

        optimizer = tf.train.AdamOptimizer(hp.learning_rate, epsilon=hp.adam_epsilon)
        update = optimizer.minimize(loss)

        merged_summary = tf.summary.merge_all()

        self.summary = merged_summary
        self.train_writer = tf.summary.FileWriter(f'tf-board/train/{time.time()}',
                                      K.get_session().graph)
        
        self.errors=tf.abs(reward_holder-responsible_weight)
        self.input_ph = t_in
        self.t_out = t_out
        self.action_ph = action_holder
        self.reward_ph = reward_holder
        self.update_op = update
        self.weights = weight_holder

        self.steps = 0


    def update(self, inputs, actions, rewards, per_weights):
        sess = K.get_session()
        _, errors, summary = sess.run([self.update_op, self.errors, self.summary], feed_dict={
            self.input_ph: inputs,
            self.action_ph:actions,
            self.reward_ph:rewards,
            self.weights:per_weights
        })
        self.train_writer.add_summary(summary, self.steps)
        self.steps += 1
        return errors

    def update_online(self):
        self.online.set_weights(self.target.get_weights())

    def save(self, file='my_model.h5'):
        self.target.save(file)

    def load_weights(self, file='my_model.h5'):
        self.online.load_weights(file)
        self.target.load_weights(file)

    def clone(self, share_online=True):
        clone = copy(self)
        if not share_online:
            clone.online, _, _ = create_model(input_shape, D)

        return clone
