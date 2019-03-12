
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from tensorflow.keras.models import load_model

from settings import s, e



choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']

# channels: arena, self, others (3), bombs, explosions, coins -> c = 8 (see get_x)
c = 6

    
def setup(agent):
    K.clear_session()
    
    D = len(choices)
    
    #========================
    #  Define Model
    #========================
    
    model = load_model('tensor_agent-model_coins-masked_actions-832k.h5')
    print('load model')

    agent.model = model

def act(agent):
    
    X = get_x(agent.game_state)
    agent.X = X

    pred = agent.model.predict(np.array([X]))
    agent.action_choice = np.argmax(pred)
    agent.next_action = choices[agent.action_choice]
    

def reward_update(agent):
    pass


def end_of_episode(agent):
    print('End of episode')


def get_x(game_state):
    arena = game_state['arena']
    self = game_state['self']
    others = game_state['others']
    bombs = game_state['bombs']
    explosions = game_state['explosions']
    coins = game_state['coins']
    # channels: arena, self, others (3), bombs, explosions, coins -> c = 8
    X = np.zeros((s.cols, s.rows, c))
    
    X[:,:,0] = arena
    
    X[self[0],self[1],1] = self[3]
    
    for i in range(len(others)):
        X[others[i][0], others[i][1], 2] = others[i][3]
    
    for i in range(len(bombs)):
        X[bombs[i][0], bombs[i][1], 3] = bombs[i][2]
    
    X[:,:,4] = explosions
    
    for i in range(len(coins)):
        X[coins[i][0], coins[i][1], 5] = 1

    return X
