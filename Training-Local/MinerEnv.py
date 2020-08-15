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
        depth = 3  # goal, energy, player's information
        goal_depth = 0
        energy_depth = 1
        players_depth = 2

        max_energy = 0
        max_goal = 67 * 50 * 4  # assume 67 steps for mining and 33 steps for relaxing

        max_x = self.state.mapInfo.max_x
        max_y = self.state.mapInfo.max_y
        max_player_energy = 50
        max_score = 67 * 50
        max_lastAction = 6
        max_status = 5

        # Building the map
        view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1, depth], dtype=float)
        for i in range(self.state.mapInfo.max_x + 1):
            for j in range(self.state.mapInfo.max_y + 1):
                view[i, j, energy_depth] = 1 / max_energy
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j, energy_depth] = 15 / max_energy  # 5~20
                elif self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j, energy_depth] = 10 / max_energy
                elif self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j, energy_depth] = 5 / max_energy  # 5, 20, 50, 100 ???????

                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j, energy_depth] = 4 / max_energy
                    view[i, j, goal_depth] = self.state.mapInfo.gold_amount(i, j) / max_goal

        # Add player's information
        index_player = 0

        view[index_player, 0, players_depth] = self.state.x / max_x
        view[index_player, 1, players_depth] = self.state.y / max_y
        view[index_player, 2, players_depth] = self.state.energy / max_player_energy
        view[index_player, 3, players_depth] = self.state.score / max_score
        if self.state.lastAction is None:
            view[index_player, 4, players_depth] = (max_lastAction + 1) / max_lastAction
        else:
            view[index_player, 4, players_depth] = self.state.lastAction / max_lastAction
        view[index_player, 5, players_depth] = self.state.status / max_status

        index_player += 1
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                view[index_player, 0, players_depth] = player["posx"] / max_x
                view[index_player, 1, players_depth] = player["posy"] / max_y
                if "energy" in player:
                    view[index_player, 2, players_depth] = player["energy"] / max_player_energy
                    view[index_player, 3, players_depth] = player["score"] / max_score
                    view[index_player, 4, players_depth] = player["lastAction"] / max_lastAction
                    view[index_player, 5, players_depth] = player["status"] / max_status
                index_player += 1

        # Convert the DQNState from list to array for training
        DQNState = np.array(view)

        return DQNState

    def get_reward(self):
        # define weight for goal and energy
        weight_goal = 0.5  # < 1
        weight_consumed_energy = 1 - weight_goal

        reward_died = -50

        # Calculate reward
        reward = 0
        score_action = self.state.score - self.score_pre
        energy_action = self.state.energy - self.energy_pre
        self.score_pre = self.state.score
        self.energy_pre = self.state.energy

        reward += score_action * weight_goal
        reward -= energy_action * weight_consumed_energy

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward = reward_died

        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward = reward_died
        # print ("reward",reward)
        return reward

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
