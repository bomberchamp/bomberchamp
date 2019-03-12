
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e

from agent_code.tensor_agent.hyperparameters import hp
from agent_code.tensor_agent.X import RelativeX2 as game_state_X

from agent_code.tensor_agent.model import FullModel
from agent_code.tensor_agent.per import PER_buffer

def augment(X, action):
    #===================
    # Data Augmentation
    #===================

    actionslr=[1,0,2,3,4,5]
    actionsud=[0,1,3,2,4,5]
    actionsudlr=[1,0,3,2,4,5]

    Xlr=np.fliplr(X)
    alr=actionslr[action]
    
    Xud=np.flipud(X)
    aud=actionsud[action]
    
    Xudlr=np.fliplr(Xud)
    audlr=actionsudlr[action]

    return [[X, action], [Xlr, alr], [Xud, aud], [Xudlr, audlr]]


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class MultiStepBuffer:
    def __init__(self):
        self.rewards=[]
        self.Xs=[]
        self.actions=[]

    def push(self, X, action_choice, reward, target_model):
        # ====
        # Multi-step learning
        # ====
        result = None

        if (len(self.rewards) >= hp.multi_step_n):
            computed_v = target_model.predict(np.array([X]))
            r = self.rewards[0] + np.max(computed_v) * hp.discount_factor ** hp.multi_step_n
            result = [self.Xs[0], self.actions[0], r]
            
            self.rewards = self.rewards[1:]
            self.actions = self.actions[1:]
            self.Xs = self.Xs[1:]
        


        self.rewards.append(0)
        self.actions.append(action_choice)
        self.Xs.append(X)

        # add gamma**0 to gamma**(n-1) times the reward to the appropriate rewards
        for i in range(len(self.rewards)):
            self.rewards[-i] += reward * hp.discount_factor ** i

        return result

    def clear(self):
        result = [self.Xs, self.actions, self.rewards]

        self.rewards=[]
        self.Xs=[]
        self.actions=[]
        return result


class TensorAgent:
    def __init__(self, input_shape, D, weights=None, model=None):
        self.input_shape = input_shape
        self.D = D

        #========================
        #  Define Model
        #========================

        if model is None:
            self.model = FullModel(input_shape, D)
        else:
            self.model = model
        
        if weights is not None:
            self.model.load_weights(weights)
        
        # Initialize all variables
        init_op = tf.global_variables_initializer()
        K.get_session().run(init_op)

        self.buffer=PER_buffer(hp.buffer_size)
        self.steps=0  #to count how many steps have been done so far

        self.ms_buffer = MultiStepBuffer()


    def act(self, X, train=False, valid_actions=None, p=None):

        if train and np.random.rand(1) <= hp.epsilon:
            if p is not None and valid_actions is not None:
                p = softmax(valid_actions * p)

            action_choice = np.random.choice(np.arange(self.D), p=p)
        else:
            pred = self.model.online.predict(np.array([X]))[0]
            if valid_actions is not None:
                pred = valid_actions * (pred - np.min(pred) + 1)

            action_choice = np.argmax(pred)

        self.steps+=1

        return action_choice


    def reward_update(self, sample):
        """
        sample: [X, action_choice, reward]
        """

        #=======================
        # Multi step learning
        # -------------------
        # Move completed samples
        # into PER buffer
        #=======================

        result = self.ms_buffer.push(*sample, target_model=self.model.target)
        if result is not None:
            X, action, reward = result
            for X_, action_ in augment(X, action):
                self.buffer.add(X_, action_, reward)

        #=======================
        # Update online network
        #=======================

        if self.steps % hp.target_network_period == 0:
            self.model.update_online()

        #=======================
        # Train target network
        #=======================

        if self.steps % hp.update_frequency == 0 and np.min(self.buffer.tree.tree[-self.buffer.tree.capacity:])>0:
            idxs, minibatch, weights = self.buffer.sample(2)
            Xs = (np.array([each[0] for each in minibatch]))
            actions = (np.array([each[1] for each in minibatch]))
            rewards = (np.array([each[2] for each in minibatch]))
            errors = self.model.update( \
                inputs = Xs, \
                actions = actions[:,None], \
                rewards = rewards[:,None], \
                per_weights = weights)

            self.buffer.update(idxs, errors)


    def end_of_episode(self, save=None):
        Xs, actions, rewards = self.ms_buffer.clear()
        for i in range(len(actions)):
            for X_, action_ in augment(Xs[i], actions[i]):
                self.buffer.add(X_, action_, rewards[i])

        if save is not None:
            self.model.save(save)

        print(f'End of episode. Steps: {self.steps}')
