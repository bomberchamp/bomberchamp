import numpy as np
import tensorflow as tf

import time
from copy import copy

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Cropping2D, Concatenate
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from agent_code.tensor_agent.hyperparameters import hp

from agent_code.tensor_agent.layers import NoisyDense, VAMerge

from agent_code.tensor_agent.names import get_random_name

writers = {}

def getFileWriter(name='train', reset=False):
    if name not in writers or reset:
        writers[name] = tf.summary.FileWriter(f'tf-board/{name}/{time.time()}')

    return writers[name]

class Counter:
    def __init__(self, count=0):
        self.count = count

    def __iadd__(self, other):
        self.count += other
        return self.count

    def __mod__(self, other):
        return self.count % other

    def __repr__(self):
        return str(self.count)

    def __int__(self):
        return self.count


def create_conv_net(shape):
    # Convolutional part of the network
    inputs = Input(shape=shape)
    x = Conv2D(32,1, padding='same', activation='relu')(inputs)
    
    x2 = Cropping2D(14)(inputs)
    x2 = Conv2D(64,3, padding='same', activation='relu')(x2)
    x2 = Flatten()(x2)
    
    x3 = Cropping2D(10)(inputs)
    x3 = Conv2D(64, 3, padding='same', activation='relu')(x3)
    x3 = Conv2D(64, 3, strides=(2,2), padding='same', activation='relu')(x3)
    x3 = Flatten()(x3)
    
    x = Conv2D(64,4, strides=(2,2), padding='same', activation='relu')(x)
    x = Conv2D(64,3, strides=(2,2), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64,3, strides=(1,1), padding='same', activation='relu')(x)
    x = Flatten()(x)

    outputs = Concatenate()([x, x2, x3])

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
    def __init__(self, input_shape, D, family=None):
        self.input_shape = input_shape
        self.D = D

        #========================
        #  Define Model
        #========================

        # Define online and target models for Double Q-learning
        self.online, o_in, o_out = create_model(input_shape, D)
        self.target, _, _ = create_model(input_shape, D)

        
        #========================
        #  Define Training Update
        #========================
        action_holder = Input(shape=(1,), dtype='int32')  # in j=0,...,D-1
        reward_holder = Input(shape=(1,))
        weight_holder = Input(shape=(1,))
        
        # applies a mask to the outputs so that only the prediction for the chosen action is considered
        responsible_weight = tf.batch_gather(o_out, action_holder)
        
        loss = tf.losses.huber_loss(reward_holder, responsible_weight, weight_holder)

        summaries = []
        if family is None:
            family = get_random_name()
        self.family = family

        summaries.append(tf.summary.scalar('loss', loss, family=family))
        summaries.append(tf.summary.scalar('reward', tf.reduce_mean(reward_holder), family=family))

        optimizer = tf.train.AdamOptimizer(hp.learning_rate, epsilon=hp.adam_epsilon)
        update = optimizer.minimize(loss)

        merged_summary = tf.summary.merge(summaries)

        self.summary = merged_summary
        self.summary_frequency = 100
        
        self.errors=tf.abs(reward_holder-responsible_weight)
        self.input_ph = o_in
        self.o_out = o_out
        self.action_ph = action_holder
        self.reward_ph = reward_holder
        self.update_op = update
        self.weights = weight_holder

        self.steps = Counter()


    def update(self, inputs, actions, rewards, per_weights):
        sess = K.get_session()
        _, errors, summary = sess.run([self.update_op, self.errors, self.summary], feed_dict={
            self.input_ph: inputs,
            self.action_ph:actions,
            self.reward_ph:rewards,
            self.weights:per_weights
        })
        if (self.steps % self.summary_frequency == 0):
            getFileWriter('train').add_summary(summary, self.steps)

        self.steps += 1
        return errors

    def update_target(self):
        self.target.set_weights(self.online.get_weights())

    def save(self, file='my_model.h5'):
        self.online.save(file)

    def load_weights(self, file='my_model.h5'):
        print('loading weights')
        self.online.load_weights(file)
        self.target.load_weights(file)
        print('weights loaded')

    def set_weights(self, weights):
        self.online.set_weights(weights)
        self.target.set_weights(weights)

    def get_weights(self):
        return self.online.get_weights()

    def clone(self, share_target=True):
        clone = copy(self)
        if not share_target:
            clone.target, _, _ = create_model(input_shape, D)

        return clone
