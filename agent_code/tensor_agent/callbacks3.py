import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e

from agent_code.tensor_agent.model import FullModel

choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']

# channels: arena, self, others (3), bombs, explosions, coins -> c = 8 (see get_x)
c = 8

class replay_buffer():
    
    def __init__(self, size = 1000, dimension=3):
        self.size = size
        self.buffer=[]
        self.dimension=dimension
        for i in range(dimension):
            self.buffer.append([])
            
    def add(self, *args):
        experience = args
        for i in range(self.dimension):
            if len(self.buffer[i])+len(experience[i])>=self.size:
                self.buffer[i]=self.buffer[i][(len(self.buffer[i])+len(experience[i]))-self.size:]
            self.buffer[i].extend(experience[i])
            
    def sample(self,batch_size):
        indexes=range(len(self.buffer[0]))
        rand=random.sample(indexes, batch_size)
        batch=[]
        for i in range(self.dimension):
            batch.append([])
            b = np.array(self.buffer[i])
            batch[i]=np.array(self.buffer[i])[rand]
        return batch     
    
    def clear(self):
        self.buffer=[]
        for i in range(self.dimension):
            self.buffer.append([])
        
    
def delayed_reward(reward, disc_factor):
    """ function that calculates delayed rewards for given list of rewards and discount_factor."""
    reward_array=np.array(reward)
    dela_rew=np.empty_like(reward)
    storage=0
    for i in range(len(reward)):
        j=len(reward)-i-1
        dela_rew[j]=storage*disc_factor+reward[j]
        storage = storage+reward[j]
    return dela_rew

def setup(agent):
    print('a')
    K.clear_session()
    
    D = len(choices)
    
    #========================
    #  Define Model
    #========================

    D = len(choices)
    input_shape = (s.cols, s.rows, c)

    model = FullModel(input_shape, D)
    
    agent.model = model
    print('a')
    
    
    # Initialize all variables
    init_op = tf.global_variables_initializer()
    K.get_session().run(init_op)

    # the alternative Keras way:
    #training_model = Model(inputs=[inputs, action_holder, reward_holder], outputs=loss)
    #training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='Adam')

    
    agent.disc_factor=0.9
    
    agent.buffer = replay_buffer() #total buffer
    agent.episode_buffer = replay_buffer() #episode buffer
    
    agent.epsilon=0.1 #for epsilon greedy policy
    agent.steps=0  #to count how many steps have been done so far

    np.random.seed()

def act(agent):
    
    X = get_x(agent.game_state)
    agent.X = X

    if np.random.rand(1) > agent.epsilon:
        pred = agent.model.online.predict(np.array([X]))
        agent.action_choice = np.argmax(pred)
        agent.next_action = choices[agent.action_choice]
    else:
        agent.next_action = np.random.choice(choices, p=[.23, .23, .23, .23, .08, .00])
    
def reward_update(agent):
    events = agent.events
    crates_destroyed = events.count(e.CRATE_DESTROYED)
    coins_found = events.count(e.COIN_FOUND)
    coins_collected = events.count(e.COIN_COLLECTED)
    opponents_killed = events.count(e.KILLED_OPPONENT)
    self_killed = events.count(e.KILLED_SELF)
    got_killed = events.count(e.GOT_KILLED)
    survived_round = events.count(e.SURVIVED_ROUND)


    # survive
    reward = -0.1 - got_killed * 100 - self_killed * 100 + 100 * survived_round
    # collect coins
    reward += 0.5 * crates_destroyed + 1 * coins_found + 10 * coins_collected
    # kill opponents
    reward += 100 * opponents_killed

    agent.reward = reward
    agent.episode_buffer.add([agent.X], [agent.action_choice], [agent.reward])
    
def end_of_episode(agent):
    #model = agent.model
    #model.train_on_batch(x, y, class_weight=None)
    x, action, reward = agent.episode_buffer.buffer
    agent.buffer.add(x, action, delayed_reward(reward, agent.disc_factor))
    agent.episode_buffer.clear() #clear episode_buffer
    agent.Xs, agent.actions, agent.rewards = agent.buffer.sample(2)
    #agent.Xs=np.array([b for b in batch[:,0]])
    #agent.actions=np.array([b for b in batch[:,1]]).reshape((-1, 1))
    #agent.rewards=np.array([b for b in batch[:,2]]).reshape((-1, 1))
    print('before update')
    agent.model.update( \
        inputs = np.array(agent.Xs), \
        actions = np.array(agent.actions)[:,None], \
        rewards = np.array(agent.rewards)[:,None])

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
        X[others[i][0], others[i][1], i+2] = others[i][3]
    
    for i in range(len(bombs)):
        X[bombs[i][0], bombs[i][1], 5] = bombs[i][2]
    
    X[:,:,6] = explosions
    
    for i in range(len(coins)):
        X[coins[i][0], coins[i][1], 7] = 1

    return X
