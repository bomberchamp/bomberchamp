from collections import namedtuple


class Hyperparameters:
    def __init__(self):
        self.buffer_size = 100000
        self.sample_size = 32
        self.discount_factor = 0.9
        self.multi_step_n = 3
        self.target_network_period = 16000
        self.epsilon = 0.
        self.learning_rate = 0.01
        self.adam_epsilon = 1.5e-4
     

        self.PER_a = 0.5
        self.PER_b = 0.4
        self.PER_anneal = (1. - 0.4) / 1000000
        self.PER_e = 1e-8
        self.rewards_update=False #include size of reward for priority update
        self.update_frequency = 4 # update every x frames

        self.peaceful = False  #bombs
        self.valid = True #If True: Invalid actions are forbidden, if false invalid actions are punished

hp = Hyperparameters()
