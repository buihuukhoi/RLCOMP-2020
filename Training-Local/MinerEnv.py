import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.score_pre = self.state.score  # Storing the last score for designing the reward function
        self.energy_pre = self.state.energy

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        depth = 3  # goal, min_energy, max_energy
        goal_depth = 0
        min_energy_depth = 1
        max_energy_depth = 1

        len_player_infor = 6 * 4

        max_goal = 67 * 50 * 4  # assume 67 steps for mining and 33 steps for relaxing
        max_energy = 100

        max_x = self.state.mapInfo.max_x
        max_y = self.state.mapInfo.max_y
        max_player_energy = 50
        max_score = 67 * 50
        max_lastAction = 6 + 1  # +1 because of None
        max_status = 5

        # Building the map
        view_1 = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, depth], dtype=float)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                view_1[i, j, min_energy_depth] = 1 / max_energy
                view_1[i, j, max_energy_depth] = 1 / max_energy

        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if obstacle["type"] == TreeID:  # Tree
                view_1[i, j, min_energy_depth] = 5 / max_energy  # 5~20
                view_1[i, j, max_energy_depth] = 20 / max_energy  # 5~20
            elif obstacle["type"] == TrapID:  # Trap
                view_1[i, j, min_energy_depth] = 10 / max_energy
                view_1[i, j, max_energy_depth] = 10 / max_energy
            elif obstacle["type"] == SwampID:  # Swamp
                view_1[i, j, min_energy_depth] = obstacle["value"] / max_energy  # 5, 20, 50, 100
                view_1[i, j, max_energy_depth] = obstacle["value"] / max_energy  # 5, 20, 50, 100

        for goal in self.state.mapInfo.golds:
            i = goal["posx"]
            j = goal["posy"]
            view_1[i, j, min_energy_depth] = 4 / max_energy
            view_1[i, j, max_energy_depth] = 4 / max_energy
            view_1[i, j, goal_depth] = goal["amount"] / max_goal

        # Add player's information
        view_2 = np.zeros([len_player_infor], dtype=float)

        index_player = 0

        view_2[index_player * 6 + 0] = self.state.x / max_x
        view_2[index_player * 6 + 1] = self.state.y / max_y
        view_2[index_player * 6 + 2] = self.state.energy / max_player_energy
        view_2[index_player * 6 + 3] = self.state.score / max_score
        if self.state.lastAction is None:  # 0 step
            view_2[index_player * 6 + 4] = max_lastAction / max_lastAction
        else:  # > 1 step
            view_2[index_player * 6 + 4] = self.state.lastAction / max_lastAction
        view_2[index_player * 6 + 5] = self.state.status / max_status

        for player in self.state.players:
            index_player += 1
            if player["playerId"] != self.state.id:
                view_2[index_player * 6 + 0] = player["posx"] / max_x
                view_2[index_player * 6 + 1] = player["posy"] / max_y
                if "energy" in player:  # > 1 step
                    view_2[index_player * 6 + 2] = player["energy"] / max_player_energy
                    view_2[index_player * 6 + 3] = player["score"] / max_score
                    view_2[index_player * 6 + 4] = player["lastAction"] / max_lastAction  # one hot
                    view_2[index_player * 6 + 5] = player["status"] / max_status
                else:  # 0 step, initial state
                    view_2[index_player * 6 + 2] = 50 / max_player_energy
                    view_2[index_player * 6 + 3] = 0 / max_score
                    view_2[index_player * 6 + 4] = max_lastAction / max_lastAction  # one hot
                    view_2[index_player * 6 + 5] = self.state.STATUS_PLAYING / max_status

        # Convert the DQNState from list to array for training
        DQNState = np.array([view_1, view_2])

        return DQNState

    def get_reward(self):
    	# return -0.01 ~ 0.01
    	# reward to go to goal
        # define weight for goal and energy
        weight_goal = 0.5  # < 1
        weight_consumed_energy = 1 - weight_goal

        max_reward = 50 - 4  # if not died
        reward_died = -100

        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre  # >= 0
        energy_action = self.state.energy - self.energy_pre  # < 0 if not relax
        self.score_pre = self.state.score
        self.energy_pre = self.state.energy

        reward += score_action * weight_goal
        reward += energy_action * weight_consumed_energy

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward = reward_died

        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward = reward_died
        # print ("reward",reward)
        return reward / max_reward / self.state.mapInfo.maxStep  # 100 steps

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
