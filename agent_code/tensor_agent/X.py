import numpy as np


from settings import s


class AbsoluteX:
    shape = (s.cols, s.rows, 8)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, self, others (3), bombs, explosions, coins -> c = 8
        X = np.zeros((s.cols, s.rows, 8))
        
        
        X[:,:,0] = arena
        
        X[self[0],self[1],1] = self[3]
        
        for i in range(len(others)):
            X[others[i][0], others[i][1], i+2] = 1 + others[i][3]
        
        for i in range(len(bombs)):
            X[bombs[i][0], bombs[i][1], 5] = bombs[i][2]
        
        X[:,:,6] = explosions
        
        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 7] = 1

        return X



class RelativeX:
    shape = (s.cols * 2 - 1, s.rows * 2 - 1, AbsoluteX.shape[2]-1)

    @staticmethod
    def get(game_state):
        X = AbsoluteX.get(game_state)
        X = np.concatenate([X[:,:,0:1], X[:,:,2:]], axis=2)

        x, y, _, _, _ = game_state['self']
        
        centered_X = np.zeros((s.cols * 2 - 1, s.rows * 2 - 1, 6))
        centered_X[s.cols-1-x:s.cols*2-1-x, s.rows-1-y:s.rows*2-1-y] = X
        return centered_X



class AbsoluteX2:
    shape = (s.cols, s.rows, 6)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, boxes, self, others, bombs, explosions, coins -> c = 7
        X = np.zeros((s.cols, s.rows, 7))
        
        
        X[:,:,0] = arena == -1

        X[:,:,1] = arena == 1
        
        X[self[0],self[1],2] = self[3]
        
        for i in range(len(others)):
            X[others[i][0], others[i][1], 3] = 1 + others[i][3]
        
        for i in range(len(bombs)):
            X[bombs[i][0], bombs[i][1], 4] = 1 - bombs[i][2] / (s.bomb_timer)
        
        X[:,:,5] = explosions
        
        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 6] = 1

        return X



class RelativeX2:
    shape = (s.cols * 2 - 1, s.rows * 2 - 1, 6)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, boxes, others, bombs, explosions, coins -> c = 6
        X = np.zeros((s.cols, s.rows, 6))
        
        
        X[:,:,0] = arena == -1

        X[:,:,1] = arena == 1
        
        for i in range(len(others)):
            X[others[i][0], others[i][1], 2] = 1 + others[i][3]
        
        for i in range(len(bombs)):
            X[bombs[i][0], bombs[i][1], 3] = 1 - bombs[i][2] / (s.bomb_timer)
        
        X[:,:,4] = explosions
        
        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 5] = 1
        
        x, y, _, _, _ = self

        centered_X = np.zeros((s.cols * 2 - 1, s.rows * 2 - 1, 6))
        centered_X[s.cols-1-x:s.cols*2-1-x, s.rows-1-y:s.rows*2-1-y] = X
        return centered_X


class Minimal:
    shape = (s.cols * 2 - 1, s.rows * 2 - 1, 2)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, boxes, others, bombs, explosions, coins -> c = 6
        X = np.zeros((s.cols, s.rows, 2))
        
        
        X[:,:,0] = arena == -1

        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 1] = 1
        
        x, y, _, _, _ = self

        centered_X = np.zeros((s.cols * 2 - 1, s.rows * 2 - 1, 2))
        centered_X[s.cols-1-x:s.cols*2-1-x, s.rows-1-y:s.rows*2-1-y] = X
        return centered_X


class SuperMinimal:
    shape = (s.cols * 2 - 1, s.rows * 2 - 1, 1)

    @staticmethod
    def get(game_state):
        arena = game_state['arena']
        self = game_state['self']
        others = game_state['others']
        bombs = game_state['bombs']
        explosions = game_state['explosions']
        coins = game_state['coins']
        # channels: arena, boxes, others, bombs, explosions, coins -> c = 6
        X = np.zeros((s.cols, s.rows, 1))

        for i in range(len(coins)):
            X[coins[i][0], coins[i][1], 0] = 1
        
        x, y, _, _, _ = self

        centered_X = np.zeros((s.cols * 2 - 1, s.rows * 2 - 1, 1))
        centered_X[s.cols-1-x:s.cols*2-1-x, s.rows-1-y:s.rows*2-1-y] = X
        return centered_X
