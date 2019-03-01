from settings import s, e
import numpy as np

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


def tile_is_free(x, y, arena, bombs, active_agents):
    is_free = (arena[x,y] == 0 and bombs[x,y] < 1)
    if is_free:
        for a in active_agents:
            a_x, a_y, _, _, _ = a
            is_free = is_free and (a_x != x or a_y != y)
    return is_free

def get_x(arena, self, others, bombs, explosions, coins):
    X = np.zeros((s.cols, s.rows, 6))
    
    X[:,:,0] = arena
    
    X[self[0],self[1],1] = 2 if self[3] > 0 else 1
    
    # one channel for each player
    #for i in range(len(others)):
    #    X[others[i][0], others[i][1], i+2] = 2 if others[i][3] > 0 else 1

    # one channel for all enemies combined
    for i in range(len(others)):
        X[others[i][0], others[i][1], 2] = 2 if others[i][3] > 0 else 1
    
    X[:,:,3] = bombs
    
    X[:,:,4] = explosions

    X[:,:,5] = coins

    return X

def play_replay(replay):
    arena = np.copy(replay['arena'])
    coins = np.zeros(arena.shape)
    coinlist = replay['coins']

    agents = [a for a in replay['agents']]
    permutations = replay['permutations']
    actions = replay['actions']

    for i in range(len(coinlist)):
        coins[coinlist[i][0], coinlist[i][1]] = 1

    bombs = np.zeros(arena.shape)
    explosions = np.zeros(arena.shape)

    Xs = []
    action_y_map = {action: i for (i, action) in enumerate(s.actions)}
    ys = []


    for i in range(replay['n_steps']):
        permutation = permutations[i]
        
        # Agents
        for j in range(len(agents)):
            agent = agents[permutation[j]]
            x, y, name, bombs_left, score = agent
            action = actions[name][i]

            X = get_x(arena, agent, [a for a in agents if a != agent], bombs, explosions, coins)
            Xs.append(X)
            ys.append(action_y_map[action])

                    
            if action == 'BOMB' and bombs_left > 0:
                bombs[x, y] = s.bomb_timer + 2
                bombs_left = -s.bomb_timer + 2
            if action == 'DOWN' and tile_is_free(x, y+1, arena, bombs, agents):
                y += 1
            if action == 'UP' and tile_is_free(x, y-1, arena, bombs, agents):
                y -= 1
            if action == 'RIGHT' and tile_is_free(x+1, y, arena, bombs, agents):
                x += 1
            if action == 'LEFT' and tile_is_free(x-1, y, arena, bombs, agents):
                x -= 1
            
            bombs_left = np.minimum(bombs_left+1, 1)
            
            agents[permutation[j]] = (x, y, name, bombs_left, score)
        
        # Coins
        for j in range(len(agents)):
            x, y, name, bombs_left, score = agents[j]
            coins[x, y] = 0
        
        # Bombs
        explosions = np.maximum(explosions, explosion_spread(bombs))
        
        bombs = np.maximum(np.zeros(bombs.shape), bombs-1)
        
        # Explosions
        arena[explosions > 1] = 0
        agents_hit = set()
        for j in range(len(agents)):
            x, y, name, bombs_left, score = agents[j]
            if explosions[x, y] > 1:
                #print(f"agent {agents[j]} was bombed at {x}, {y} in step {i}")
                agents_hit.add(agents[j])
        
        explosions = np.maximum(np.zeros(explosions.shape), explosions-1)
        
        for a in agents_hit:
            agents.remove(a)

    return Xs, ys
