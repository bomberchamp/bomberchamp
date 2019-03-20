from settings import s
from settings import e as ev
import numpy as np
import pickle

import random

from copy import copy

##TODO get times for every step and use mean for computation of reward slow

class expl():
    def __init__(self, owner, coords, bomb):
        self.owner=owner
        self.timer=s.bomb_timer + 1
        self.coords=coords
        self.bomb=bomb
        
    def dist(self, otherbomb):
        return np.abs(np.array(self.bomb)-np.array(otherbomb))
        
        
def explosion_spread(bombs):
    xs, ys = np.where(bombs == 1)
    
    bomb_power = s.bomb_power

    x_odd = (xs & 1) == 1
    y_odd = (ys & 1) == 1

    x_spread = (np.arange(bomb_power*2 + 1) - bomb_power)[:,None] + (xs[y_odd])[None,:]
    y_spread = (np.arange(bomb_power*2 + 1) - bomb_power)[:,None] + (ys[x_odd])[None,:]

    x_spread = np.maximum(np.minimum(x_spread, s.cols-2), 1)
    y_spread = np.maximum(np.minimum(y_spread, s.rows-2), 1)

    x_spread_y = np.zeros(x_spread.shape) + (ys[y_odd])[None,:]
    x_spread = x_spread.flatten()
    x_spread_y = x_spread_y.flatten()

    y_spread_x = np.zeros(y_spread.shape) + (xs[x_odd])[None,:]
    y_spread = y_spread.flatten()
    y_spread_x = y_spread_x.flatten()

    new_explosion_spread = np.zeros(bombs.shape) - 3
    new_explosion_spread[x_spread.astype(int), x_spread_y.astype(int)] = s.explosion_timer + 1
    new_explosion_spread[y_spread_x.astype(int), y_spread.astype(int)] = s.explosion_timer + 1
    return new_explosion_spread

def explosion_spread_xy(x, y):
    xs, ys = np.array([x]) , np.array([y])

    bomb_power = s.bomb_power

    x_odd = (xs & 1) == 1
    y_odd = (ys & 1) == 1

    x_spread = (np.arange(bomb_power*2 + 1) - bomb_power)[:,None] + (xs[y_odd])[None,:]
    y_spread = (np.arange(bomb_power*2 + 1) - bomb_power)[:,None] + (ys[x_odd])[None,:]

    x_spread = np.maximum(np.minimum(x_spread, s.cols-2), 1)
    y_spread = np.maximum(np.minimum(y_spread, s.rows-2), 1)

    x_spread_y = np.zeros(x_spread.shape) + (ys[y_odd])[None,:]
    x_spread = x_spread.flatten()
    x_spread_y = x_spread_y.flatten()

    y_spread_x = np.zeros(y_spread.shape) + (xs[x_odd])[None,:]
    y_spread = y_spread.flatten()
    y_spread_x = y_spread_x.flatten()
    elements=set()
    for i in range(len(x_spread)):
        if (x_spread[i], x_spread_y[i]) not in elements:
            elements.add((int(x_spread[i]), int(x_spread_y[i])))
    for i in range(len(y_spread)):
        if (y_spread_x[i], y_spread[i]) not in elements:
            elements.add((int(y_spread_x[i]), int(y_spread[i])))

    return elements


def play_replay(replay, get_x, action_y_map, **kwargs):
    arena = np.copy(replay['arena'])
    coins = np.zeros(arena.shape)
    coinlist = replay['coins']

    agents = [a for a in replay['agents']]
    permutations = replay['permutations']
    actions = replay['actions']

    for i in range(len(coinlist)):
        coins[coinlist[i][0], coinlist[i][1]] = 1

    Xs = []
    ys = []
    rs = []
    agent_assignment = []

    game = Game(arena, coins, agents, **kwargs)
    
    for i in range(replay['n_steps']):
        permutation = permutations[i]
        agent_Xs = {}
        agent_actions = {}

        for agent in game.agents:
            _, _, name, _, _ = agent
            agent_Xs[name] = get_x(game.get_game_state(agent))
            agent_actions[name] = actions[name][i]


        rewards, _ = game.step(agent_actions, permutation)

        for name in agent_actions.keys():
            Xs.append(agent_Xs[name])
            ys.append(action_y_map[agent_actions[name]])
            rs.append(rewards[name])
            agent_assignment.append(name)
    
    #print(game.score)
    return Xs, ys, rs, agent_assignment


def get_valid_actions(x, y, b, game):
    # choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
    valid = np.ones((6))
    if not game.tile_is_free(x, y-1):
        valid[2] = 0 # UP invalid
    if not game.tile_is_free(x, y+1):
        valid[3] = 0 # DOWN invalid
    if not game.tile_is_free(x-1, y):
        valid[1] = 0 # LEFT invalid
    if not game.tile_is_free(x+1, y):
        valid[0] = 0 # RIGHT invalid
    if b<1:
        valid[4] = 0

    return valid


class Game:
    def __init__(self, arena, coins, agents, aux_reward_crates=0):
        self.arena = np.copy(arena)
        self.coins = np.copy(coins)
        self.agents = copy(agents)

        self.bombs = np.zeros(arena.shape)
       
        self.explosions = np.zeros(arena.shape)

        self.exp=[]
        self.score=dict()

        for agent in agents:
            _, _, name, _, _ = agent
            self.score[name]=0

        self.steps = 0
        self.terminated = False

    @staticmethod
    def create_arena(agent_names, crate_density=s.crate_density):
        # Arena with wall and crate layout
        arena = (np.random.rand(s.cols, s.rows) < crate_density).astype(int)
        arena[:1, :] = -1
        arena[-1:,:] = -1
        arena[:, :1] = -1
        arena[:,-1:] = -1
        for x in range(s.cols):
            for y in range(s.rows):
                if (x+1)*(y+1) % 2 == 1:
                    arena[x,y] = -1

        # Starting positions
        start_positions = [(1,1), (1,s.rows-2), (s.cols-2,1), (s.cols-2,s.rows-2)]
        random.shuffle(start_positions)
        for (x,y) in start_positions:
            for (xx,yy) in [(x,y), (x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                if arena[xx,yy] == 1:
                    arena[xx,yy] = 0

        # Distribute coins evenly
        coins = np.zeros(arena.shape)


        for i in range(3):
            for j in range(3):
                n_crates = (arena[1+5*i:6+5*i, 1+5*j:6+5*j] == 1).sum()
                while True:
                    x, y = np.random.randint(1+5*i,6+5*i), np.random.randint(1+5*j,6+5*j)
                    if (n_crates == 0 and arena[x,y] == 0) or arena[x,y] == 1:
                        coins[x,y] = 1
                        break

        # Distribute starting positions
        agents = []
        for name in agent_names:
            x, y = start_positions.pop()
            agents.append((x, y, name, 1, 0))

        return [arena, coins, agents]

    def step(self, agent_actions, permutation=None):
        self.steps += 1

        if permutation is None:
            permutation = np.random.permutation(len(self.agents))

        step_score = {n: 0 for n in self.score.keys()}
        events = {n: {event: 0 for event in ev} for n in self.score.keys()}

        # Agents
        for j in range(len(self.agents)):
            agent = self.agents[permutation[j]]
            x, y, name, bombs_left, score = agent
            action = agent_actions[name]

                    
            if action == 'BOMB' and bombs_left > 0:
                self.bombs[x, y] = s.bomb_timer + 2
                bombs_left = -s.bomb_timer - 1
                self.exp.append(expl(agent,explosion_spread_xy(x,y),(x,y)))
                events[name][ev.BOMB_DROPPED] += 1
            elif action == 'DOWN' and self.tile_is_free(x, y+1):
                y += 1
                events[name][ev.MOVED_DOWN] += 1
            elif action == 'UP' and self.tile_is_free(x, y-1):
                y -= 1
                events[name][ev.MOVED_UP] += 1
            elif action == 'RIGHT' and self.tile_is_free(x+1, y):
                x += 1
                events[name][ev.MOVED_RIGHT] += 1
            elif action == 'LEFT' and self.tile_is_free(x-1, y):
                x -= 1
                events[name][ev.MOVED_LEFT] += 1
            elif action == 'WAIT':
                events[name][ev.WAITED] += 1
            else:
                events[name][ev.INVALID_ACTION] += 1
            
            bombs_left = np.minimum(bombs_left+1, 1)
            
            self.agents[permutation[j]] = (x, y, name, bombs_left, score)

        
        for j in range(len(self.agents)):
            x, y, name, bombs_left, score = self.agents[j]
            if self.coins[x,y]==1:
                step_score[name]+=s.reward_coin
                events[name][ev.COIN_COLLECTED] += 1
            self.coins[x,y]=0
        
            
        
            
        
        # Bombs
        self.explosions = np.maximum(self.explosions, explosion_spread(self.bombs))
        
        self.bombs = np.maximum(np.zeros(self.bombs.shape), self.bombs-1)
        
        # Explosions
        boxes_hit = np.logical_and(self.explosions > 1, self.arena == 1)
        self.arena[self.explosions > 1] = 0


        for e in self.exp:
            box_count = np.sum(boxes_hit[tuple(zip(*e.coords))])
            _, _, name, _, _ = e.owner
            events[name][ev.CRATE_DESTROYED] += box_count


        agents_hit = set()
        for j in range(len(self.agents)):
            x, y, name, bombs_left, score = self.agents[j]
            if self.explosions[x, y] > 1:
                #print(f"agent {self.agents[j]} was bombed at {x}, {y} in step {self.steps}")
                owners=[]
                dists=[]
                for e in self.exp:
                    if (x,y) in e.coords and e.timer<=0:
                        owners.append(e)
                        dists.append(e.dist((x,y)))
                if len(owners)==1:
                    killer=owners[0].owner
                else:
                    killer=owners[np.argmin(np.array(dists))].owner
                a, b, name_k, c, d = killer
                if name_k!=name:
                    #print('bombed by', name_k)
                    step_score[name_k]+=s.reward_kill
                    events[name_k][ev.KILLED_OPPONENT] += 1
                else:
                    #print('suicide')
                    events[name_k][ev.KILLED_SELF] += 1
                agents_hit.add(self.agents[j])


        self.explosions = np.maximum(np.zeros(self.explosions.shape), self.explosions-1)
        
        exp_to_remove = []
        for e in self.exp:
            e.timer -= 1
            if e.timer < -1:
                exp_to_remove.append(e)

        for e in exp_to_remove:
            self.exp.remove(e)


        for a in agents_hit:
            _, _, n, _, _ = a
            events[n][ev.GOT_KILLED] += 1
            self.agents.remove(a)

        if len(self.agents) == 0 or self.steps >= 400 or (np.all(self.arena != 1) and np.all(self.coins == 0)):
            self.terminated = True

        for _, _, n, _, _ in self.agents:
            self.score[n] += step_score[n]

        return step_score, events


    def get_state(self, agent):
        # deprecated, but still used by replay
        return [self.arena, agent, [a for a in self.agents if a != agent], self.bombs, self.explosions, self.coins]

    def get_game_state(self, agent):
        # mimics format of original framework
        return {
            'step': self.step,
            'arena': self.arena,
            'self': agent,
            'others': [a for a in self.agents if a != agent],
            'bombs': np.concatenate([np.stack(np.where(self.bombs >= 1)),[self.bombs[self.bombs >= 1] - 1]]).T.astype(int),
            'explosions': self.explosions,
            'coins': np.array(np.where(np.logical_and(self.coins == 1, self.arena == 0))).T
        }

    def tile_is_free(self, x, y):
        is_free = (self.arena[x,y] == 0 and self.bombs[x,y] < 1)
        if is_free:
            for a in self.agents:
                a_x, a_y, _, _, _ = a
                is_free = is_free and (a_x != x or a_y != y)
        return is_free
