#from warnings import simplefilter
#simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Input, Concatenate, concatenate
from keras import optimizers
from keras import backend as K
import tensorflow as tf
from random import random, randrange

class DQN:
    def __init__(
            self,
            input_shape_1=(21, 9, 2),  # The number of inputs for the DQN network
            input_shape_2=(24,),
            action_space=6,  # The number of actions for the DQN network
            gamma=0.99,  # The discount factor
            #gamma=0.9
            epsilon=1,  # Epsilon - the exploration factor
            epsilon_min=0.01,  # The minimum epsilon
            epsilon_decay=0.99999,  # The decay epsilon for each update_epsilon time
            learning_rate=0.00025,  # The learning rate for the DQN network
            tau=0.125,  # The factor for updating the DQN target network from the DQN network
            sess=None,
    ):
        self.input_shape_1 = input_shape_1
        self.input_shape_2 = input_shape_2
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
        x1 = Input(shape=self.input_shape_1, name="state_map")
        conv = Conv2D(256, (3, 3), activation="relu")(x1)
        conv = Conv2D(256, (3, 3), activation="relu")(conv)
        flatten_1 = Flatten()(conv)

        x2 = Input(shape=self.input_shape_2, name="state_users")
        #flatten_2 = Flatten()(x2)

        #concat = Concatenate()([flatten_1, flatten_2])
        concat = concatenate([flatten_1, x2])
        d = Dense(64, activation='relu')(concat)
        d = Dense(self.action_space, activation="linear")(d)

        model = Model(inputs=[x1, x2], outputs=d)
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        #print(model.summary())
        return model

        """
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=self.input_shape))
        model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        #model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        #model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64, activation="relu"))

        model.add(Dense(self.action_space, activation="linear"))
        # sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
        # model.compile(optimizer=sgd, loss='mse')
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=0.0001), metrics=['accuracy'])
        return model
        """

    def get_qs(self, state_map, state_users):
        # check shape again ??????????????????????????
        return self.model.predict({"state_map": state_map.reshape(1, 21, 9, 3), "state_users": state_users.reshape(1, 24)})

    def act(self, state_map, state_users):
        # Get the index of the maximum Q values
        if random() < self.epsilon:
            action = randrange(self.action_space)
        else:
            action = np.argmax(self.get_qs(state_map, state_users))
        return action

    def replay(self, samples, batch_size):
        # samples are taken randomly in Memory.sample()
        inputs_map = np.zeros((batch_size, *(self.input_shape_1)))
        inputs_users = np.zeros((batch_size, *(self.input_shape_2)))
        targets = np.zeros((batch_size, self.action_space))

        for i in range(0, batch_size):
            state_map = samples[0][i]
            state_users = samples[1][i]
            action = samples[2][i]
            reward = samples[3][i]
            new_state_map = samples[4][i]
            new_state_users = samples[5][i]
            done = samples[6][i]

            # inputs[i] = state
            # check input shape again ?????????????????????????????????????????????????????????
            targets[i, :] = self.target_model.predict({"state_map": state_map.reshape(1, 21, 9, 3), "state_users": state_users.reshape(1, 24)})
            # targets[i, :] = self.get_qs(state)
            if done:
                targets[i, action] = reward  # if terminated ==> no new_state ==> only equals reward
            else:
                # check input shape again ?????????????????????????????????????????????????????????
                max_future_qs = np.max(self.target_model.predict({"state_map": new_state_map.reshape(1, 21, 9, 3), "state_users": new_state_users.reshape(1, 24)}))
                targets[i, action] = reward + self.gamma * max_future_qs
        # Training
        loss = self.model.train_on_batch({"state_map": inputs_map, "state_users": inputs_users}, targets)

    def update_target_model(self):
        weights = self.model.get_weights()
        #target_weights = self.target_model.get_weights()
        #for i in range(0, len(target_weights)):
        #    target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        #self.target_model.set_weights(target_weights)
        self.target_model.set_weights(weights)

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
