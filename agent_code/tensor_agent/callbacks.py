
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e

from agent_code.tensor_agent.hyperparameters import hp

from agent_code.tensor_agent.model import FullModel
from agent_code.tensor_agent.per import PER_buffer

choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']

# channels: arena, self, others (3), bombs, explosions, coins -> c = 8 (see get_x)
c = 8

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

    D = len(choices)
    input_shape = (s.cols, s.rows, c)
    model = FullModel(input_shape, D)
    agent.model = model
    
    agent.model.load_weights('tensor_agent-model.h5')
    
    # Initialize all variables
    init_op = tf.global_variables_initializer()
    K.get_session().run(init_op)

    # the alternative Keras way:
    #training_model = Model(inputs=[inputs, action_holder, reward_holder], outputs=loss)
    #training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='Adam')
    
    agent.buffer=PER_buffer(hp.buffer_size,0.5,0.1,0.1,0.1)   #(hp.buffer_size, PER_a, PER_b, PER_e, anneal)
    agent.steps=0  #to count how many steps have been done so far

    agent.rewards=[]
    agent.Xs=[]
    agent.actions=[]
    np.random.seed()

def filter_invalid(game_state, p):
    # choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
    valid = np.ones((6))
    x, y, _, b, _ = game_state['self']
    arena = game_state['arena']
    if not tile_is_free(x, y-1, game_state):
        valid[2] = 0 # UP invalid
    if not tile_is_free(x, y+1, game_state):
        valid[3] = 0 # DOWN invalid
    if not tile_is_free(x-1, y, game_state):
        valid[1] = 0 # LEFT invalid
    if not tile_is_free(x+1, y, game_state):
        valid[0] = 0 # RIGHT invalid
    if b <= 0:
        valid[4] = 0

    valid[4] = 0
    x = valid * p
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def tile_is_free(x, y, game_state):
    is_free = game_state['arena'][x,y] == 0
    if is_free:
        for obstacle in game_state['bombs']:
            o_x, o_y, _ = obstacle
            is_free = is_free and (o_x != x or o_y != y)
    return is_free

def act(agent):
    
    X = get_x(agent.game_state)
    agent.X = X

    if  np.random.rand(1) > hp.epsilon:
        pred = agent.model.online.predict(np.array([X]))
        agent.action_choice = np.argmax(filter_invalid(agent.game_state, pred[0]))
        if (choices[agent.action_choice] == 'BOMB'): # for collecting coins
            agent.action_choice = 5
            #agent.action_choice = np.random.choice(np.arange(len(choices)), p=filter_invalid(agent.game_state, [.23, .23, .23, .23, .00, .08]))
        agent.next_action = choices[agent.action_choice]
    else:
        agent.action_choice = np.random.choice(np.arange(len(choices)), p=filter_invalid(agent.game_state, [.23, .23, .23, .23, .00, .08])) # coins
        #agent.action_choice = np.random.choice(np.arange(len(choices)), p=filter_invalid(agent.game_state, [.23, .23, .23, .23, .08, .00]))
        if (choices[agent.action_choice] == 'BOMB'): # for collecting coins
            agent.action_choice = 5
        agent.next_action = choices[agent.action_choice]
    agent.steps+=1
        
    
def end_of_episode(agent):
    #model = agent.model
    #model.train_on_batch(x, y, class_weight=None)
    #agent.rewards=delayed_reward(agent.rewards,hp.discount_factor)
    for i in range(len(agent.actions)):
        agent.buffer.add([agent.Xs[i]], [agent.actions[i]], [agent.rewards[i]])

    agent.rewards=[]
    agent.Xs=[]
    agent.actions=[]

    agent.model.save('tensor_agent-model.h5')
    print(f'End of episode. Steps: {agent.steps}')
    
    
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
    reward = -0.1 - got_killed * s.reward_kill - self_killed * s.reward_kill #+ 100 * survived_round
    # collect coins
    reward += 0.1 * crates_destroyed + 0.5 * coins_found + s.reward_coin * coins_collected
    # kill opponents
    reward += s.reward_kill * opponents_killed

    agent.reward = reward


    # ====
    # Multi-step learning
    # ====


    if (len(agent.rewards) >= hp.multi_step_n):
        computed_v = agent.model.target.predict(np.array([agent.X]))
        r = agent.rewards[0] + np.max(computed_v)
        agent.buffer.add([agent.Xs[0]], [agent.actions[0]], [r])

        agent.rewards = agent.rewards[1:]
        agent.actions = agent.actions[1:]
        agent.Xs = agent.Xs[1:]


    agent.rewards.append(0)
    agent.actions.append(agent.action_choice)
    agent.Xs.append(agent.X)

    # add gamma**0 to gamma**(n-1) times the reward to the appropriate rewards
    for i in range(len(agent.rewards)):
        agent.rewards[-i] += reward * hp.discount_factor ** i

    if agent.steps % hp.target_network_period == 0:
        agent.model.update_online()


    if agent.steps % 10 == 0 and np.min(agent.buffer.tree.tree[-agent.buffer.tree.capacity:])>0:
        idxs, minibatch, weights = agent.buffer.sample(2)
        Xs = np.concatenate(np.array([each[0] for each in minibatch]))
        actions = np.concatenate(np.array([each[1] for each in minibatch]))
        rewards = np.concatenate(np.array([each[2] for each in minibatch]))
        errors = agent.model.update( \
            inputs = Xs, \
            actions = actions[:,None], \
            rewards = rewards[:,None], \
            per_weights = weights)

        agent.buffer.update(idxs, errors)


