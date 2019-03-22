from agent_code.simple_agent import callbacks
from agent_code.tensor_agent.agent import TensorAgent
from types import SimpleNamespace


choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']
action_y_map = {choices[i]: i for i in range(len(choices))}

class SimpleAgent(TensorAgent):
    def __init__(self, *args, **kwargs):
        TensorAgent.__init__(self, *args, **kwargs)

        self.logger = SimpleNamespace(name='SimpleAgentLogger')
        self.logger.debug = lambda x: x
        self.logger.info = lambda x: x
        callbacks.setup(self)

    def act(self, X, train=False, valid_actions=None, p=None, game_state=None):
        self.game_state = {
            'step': game_state['step'],
            'arena': game_state['arena'],
            'self': game_state['self'],
            'others': game_state['others'],
            'bombs': game_state['bombs'].tolist(),
            'explosions': game_state['explosions'],
            'coins': game_state['coins'].tolist()
        }

        callbacks.act(self)
        self.steps+=1

        return action_y_map[self.next_action]
