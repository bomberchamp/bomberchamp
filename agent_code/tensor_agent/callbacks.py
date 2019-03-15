
import numpy as np

from tensorflow.keras import backend as K

from settings import s, e

from agent_code.tensor_agent.hyperparameters import hp
from agent_code.tensor_agent.X import Minimal as game_state_X

from agent_code.tensor_agent.agent import TensorAgent

choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']

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


def get_valid_actions(game_state):
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

    return valid

def tile_is_free(x, y, game_state):
    is_free = game_state['arena'][x,y] == 0
    if is_free:
        for obstacle in game_state['bombs']:
            o_x, o_y, _ = obstacle
            is_free = is_free and (o_x != x or o_y != y)
    return is_free




def setup(self):

    K.clear_session()
    
    D = len(choices)
    
    self.ta = TensorAgent(game_state_X.shape, D, weights=None)


def act(self):
    train = self.game_state['train']

    X = game_state_X.get(self.game_state)
    self.X = X

    valid_actions = get_valid_actions(self.game_state)

    self.action_choice = self.ta.act(X, train=train, valid_actions=valid_actions, p=[0.23, 0.23, 0.23, 0.23, 0., 0.8])

    if hp.peaceful and choices[self.action_choice] == 'BOMB':
        self.action_choice = 5

    self.next_action = choices[self.action_choice]
    
    
def end_of_episode(self):
    self.ta.end_of_episode()
    

def reward_update(self):
    events = self.events
    crates_destroyed = events.count(e.CRATE_DESTROYED)
    coins_found = events.count(e.COIN_FOUND)
    coins_collected = events.count(e.COIN_COLLECTED)
    opponents_killed = events.count(e.KILLED_OPPONENT)
    self_killed = events.count(e.KILLED_SELF)
    got_killed = events.count(e.GOT_KILLED)
    survived_round = events.count(e.SURVIVED_ROUND)
    invalid_actions = events.count(e.INVALID_ACTION)


    # survive
    reward = -0.1 - got_killed * s.reward_kill - self_killed * s.reward_kill #+ 100 * survived_round
    # collect coins
    reward += 0.1 * crates_destroyed + 0.5 * coins_found + s.reward_coin * coins_collected
    # kill opponents
    reward += s.reward_kill * opponents_killed

    if hp.valid==False:
        reward -= invalid_actions * 0.2

    self.reward = reward

    self.ta.reward_update([self.X, self.action_choice, self.reward])
