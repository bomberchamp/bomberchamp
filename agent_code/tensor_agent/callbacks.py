
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras import backend as K

from settings import s, e



choices = ['RIGHT', 'LEFT', 'UP', 'DOWN', 'BOMB', 'WAIT']


def setup(agent):
    K.clear_session()
    
    D = len(choices)
    
    #========================
    #  Define Model
    #========================
    
    inputs = Input(shape=(1,))
    x = Dense(64, activation='relu')(inputs)
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
    responsible_weight = tf.reduce_sum(tf.boolean_mask(pred, tf.one_hot(action_holder, D)[:,0,:]))

    loss = - (tf.log(responsible_weight) * reward_holder)

    optimizer = tf.train.AdamOptimizer(0.1)
    update = optimizer.minimize(loss)
    
    
    # Initialize all variables
    init_op = tf.global_variables_initializer()
    K.get_session().run(init_op)

    # the alternative Keras way:
    #training_model = Model(inputs=[inputs, action_holder, reward_holder], outputs=loss)
    #training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer='Adam')

    
    agent.update = update
    
    agent.inputs = inputs
    agent.action_holder = action_holder
    agent.reward_holder = reward_holder
    

    np.random.seed()

def act(agent):
    # agent.game_state
    print('Pick action at random')

    #agent.next_action = np.random.choice(choices, p=[.23, .23, .23, .23, .08, .00])

    pred = agent.model.predict(np.array([1]))
    agent.next_action = choices[np.argmax(pred)]
    print(agent.next_action)

def reward_update(agent):
    print('Update')
    # agent.events
    pass

def end_of_episode(agent):
    #model = agent.model
    #model.train_on_batch(x, y, class_weight=None)
    
    sess = K.get_session()
    sess.run([agent.update], feed_dict={agent.inputs: [[1]], agent.reward_holder:[[2]],agent.action_holder:[[5]]})
    print('End of Episode')
    pass
