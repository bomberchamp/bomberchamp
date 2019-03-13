# in reference to https://gist.github.com/simoninithomas/d6adc6edb0a7f37d6323a5e3d2ab72ec#file-dueling-deep-q-learning-with-doom-double-dqns-and-prioritized-experience-replay-ipynb

import numpy as np

from agent_code.tensor_agent.hyperparameters import hp

class SumTree(object):
    def __init__(self, capacity): #initalize tree and data with only zeros
        
        self.capacity=capacity
        self.tree=np.zeros(2*capacity-1)#actual tree
        self.data=np.zeros(capacity, dtype=object)#here the data is stored
        self.pointer=0
        self.total_priority=0.1   #for normalization get priority of root

        
    def update(self, index, priority):
        old_priority=self.tree[index]
        self.tree[index]=priority
        
        #update upper parent nodes
        while index!=0: #as long as root is not reached
            
            index=int(np.floor((index-1)/2))
            self.tree[index]= self.tree[index]+priority-old_priority
            
    def add(self, priority, new_data):
        
        self.data[self.pointer]=new_data
        t_index=self.pointer+self.capacity-1
        self.update(t_index, priority)
            
        self.pointer+=1
        if self.pointer>=self.capacity:
            self.pointer=0
        
        self.total_priority=self.tree[0]
        
    def get_leave(self, value):
        bottom=False
        parent=0
        
        while not bottom:
            
            left_child=parent*2+1
            right_child=left_child+1
            if left_child>=len(self.tree):
                break
            
            if self.tree[left_child]>=value:
                parent=left_child
            else:
                value-=left_child
                parent=right_child
        leave_index=parent
        return leave_index, self.tree[leave_index], self.data[leave_index-self.capacity+1]
            

            
class PER_buffer(object):
    
    def __init__(self, capacity):
    
        self.capacity=capacity   
        self.tree=SumTree(capacity)
        self.default_max_p=1.
        
    def add(self, *args):
        experience = args
        #new experience gets maximum priority
        max_p=np.max(self.tree.tree[-self.tree.capacity:]) #search for maximal priority in leave nodes
        
        if max_p==0:                        #priority can't be zero because then experience would never be picked
            max_p=self.default_max_p
            
        self.tree.add(max_p, experience)
        
        
        
    def sample(self, k):  #k:how many experiences in one sample
        
        priority_range = self.tree.total_priority/k
        minibatch=[]
        idxs = np.ones((k,), dtype=np.int32)
        weights = np.ones((k, 1))
        #to normalize the weights, the maximal weight needs to be calculated
        max_weight=1/(k*np.min(self.tree.tree[-self.tree.capacity:]))**hp.PER_b
        
        for i in range(k):
            lower_bound = priority_range*i
            upper_bound=lower_bound+priority_range
            
            #now get a random sample from that range
            value=np.random.uniform(lower_bound,upper_bound)
            leave_index, value_priority, value_data = self.tree.get_leave(value)
            idxs[i]=leave_index            
            prob_weight=value_priority/self.tree.total_priority
            weights[i,0]=(1/(k*prob_weight)**hp.PER_b)/max_weight
            minibatch.append(value_data)
        hp.PER_b=np.minimum(1., hp.PER_b+hp.PER_anneal)
        return idxs, minibatch, weights
    
    def update(self, idxs, errors, rewards=None):
        ''' It is important to use tree idx here, not tree '''
        
        if hp.rewards_update==False:
            priorities=errors+hp.PER_e
        else:
            priorities=errors+hp.PER_e+np.maximum(rewards, 0.5)
        priorities=np.minimum(priorities, self.default_max_p)
        pri_a=priorities**hp.PER_a   #modified priority that is actually used
        for i, p in zip(idxs, pri_a):
            self.tree.update(i, p)
