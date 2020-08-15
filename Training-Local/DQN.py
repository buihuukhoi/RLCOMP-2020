#from warnings import simplefilter
#simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from random import random, randrange

class DQN:
    def __init__(
            self,
            input_shape=(21, 9, 3),  # The number of inputs for the DQN network
            action_space=6,  # The number of actions for the DQN network
            gamma=0.99,  # The discount factor
            epsilon=1,  # Epsilon - the exploration factor
            epsilon_min=0.01,  # The minimum epsilon
            epsilon_decay=0.999,  # The decay epsilon for each update_epsilon time
            learning_rate=0.00025,  # The learning rate for the DQN network
            tau=0.125,  # The factor for updating the DQN target network from the DQN network
            sess=None,
    ):
        self.input_shape = input_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        # create main model
        self.model = self.create_model()
        # create target model
        self.target_model = self.create_model()

        # Tensorflow GPU optimization
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        K.set_session(sess)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(self.action_space, activation="linear"))
        # sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        # model.compile(optimizer=sgd, loss='mse')
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.001), metrics=['accuracy'])
        return model

    def get_qs(self, state):
        # check shape again ??????????????????????????
        return self.model.predict(state)

    def act(self, state):
        # Get the index of the maximum Q values
        if random() < self.epsilon:
            action = randrange(self.action_space)
        else:
            action = np.argmax(self.get_qs(state))
        return action

    def replay(self, samples, batch_size):
        # samples are taken randomly in Memory.sample()
        inputs = np.zeros((batch_size, self.input_shape))
        targets = np.zeros((batch_size, self.action_space))

        for i in range(0, batch_size):
            state = samples[0][i, :]
            action = samples[1][i]
            reward = samples[2][i]
            new_state = samples[3][i, :]
            done = samples[4][i]

            inputs[i, :] = state
            # check input shape again ?????????????????????????????????????????????????????????
            targets[i, :] = self.target_model.predict(state)
            # targets[i, :] = self.get_qs(state)
            if done:
                targets[i, action] = reward  # if terminated ==> no new_state ==> only equals reward
            else:
                # check input shape again ?????????????????????????????????????????????????????????
                max_future_qs = np.max(self.target_model.predict(new_state))
                targets[i, action] = reward + self.gamma * max_future_qs
        # Training
        loss = self.model.train_on_batch(inputs, targets)

    def update_target_model(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(0, len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def update_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    # check again ??????????????????????????????????????????????????????????????????
    def save_model(self, path, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(path + model_name + ".h5")
            print("Saved model to disk")
