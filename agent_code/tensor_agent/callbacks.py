
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e



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
    K.clear_session()
    
    D = len(choices)
    
    #========================
    #  Define Model
    #========================
    
    inputs = Input(shape=(s.cols, s.rows, c))
    x = Conv2D(16, 3)(inputs)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    pred = Dense(D, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=pred)
    #model.compile(loss="hinge", optimizer="adam")

    agent.model = model

    
    #========================
    #  Define Training Update
    #========================

    action_holder = Input(shape=(1,), dtype='int32')  # in j=0,...,D-1
    reward_holder = Input(shape=(1,))
    
    # applies a mask to the outputs so that only the prediction for the chosen action is considered
    responsible_weight = tf.gather(pred, action_holder, axis=1)

    loss = - tf.reduce_mean(tf.log(responsible_weight) * reward_holder)

    optimizer = tf.train.AdamOptimizer(0.1)
    update = optimizer.minimize(loss)
    
    
    # Initialize all variables
    init_op = tf.global_variables_initializer()
    K.get_session().run(init_op)

    # the alternative Keras way:
    #training_model = Model(inputs=[inputs, action_holder, reward_holder], outputs=loss)
    #training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='Adam')

    
    agent.update = update
    agent.disc_factor=0.7
    agent.inputs = inputs
    agent.action_holder = action_holder
    agent.reward_holder = reward_holder
    
    agent.buffer = replay_buffer() #total buffer
    agent.episode_buffer = replay_buffer() #episode buffer
    
    agent.epsilon=0.1 #for epsilon greedy policy
    agent.steps=0  #to count how many steps have been done so far

    np.random.seed()

def act(agent):
    
    X = get_x(agent.game_state)
    agent.X = X

    if np.random.rand(1) > agent.epsilon:
        pred = agent.model.predict(np.array([X]))
        agent.action_choice = np.argmax(pred)
        agent.next_action = choices[agent.action_choice]
    else:
        agent.next_action = np.random.choice(choices, p=[.23, .23, .23, .23, .08, .00])
    
def reward_update(agent):
    events = agent.events
    reward = -1
    reward += events.count(e.COIN_FOUND)
    reward += events.count(e.COIN_COLLECTED)
    reward += 2 * events.count(e.KILLED_OPPONENT)
    reward -= 10 * events.count(e.KILLED_SELF)
    reward -= 5 * events.count(e.GOT_KILLED)
    reward += 20 * events.count(e.SURVIVED_ROUND)
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
    sess = K.get_session()
  
    sess.run([agent.update], feed_dict={agent.inputs: np.array(agent.Xs),  agent.reward_holder:np.array(agent.rewards)[:,None],agent.action_holder:np.array(agent.actions)[:,None]}) 
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
