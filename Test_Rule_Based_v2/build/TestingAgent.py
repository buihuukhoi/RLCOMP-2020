from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import sys
from MINER_STATE import State
from GAME_SOCKET import GameSocket


class MyBot:
    ACTION_GO_LEFT = 0
    ACTION_GO_RIGHT = 1
    ACTION_GO_UP = 2
    ACTION_GO_DOWN = 3
    ACTION_FREE = 4
    ACTION_CRAFT = 5

    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()

        self.is_moving_right = True  # default: go to right side
        self.steps = 0
        self.pre_action = 0

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

    def step(self, tmp_action):  # step process
        self.socket.send(tmp_action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING

    def goLeftOrRight(self, my_bot_x, my_bot_y, initial_flag=False):
        total_gold_left = 0
        total_gold_right = 0
        for gold in self.state.mapInfo.golds:
            gold_amount = gold["amount"]
            if gold_amount > 0:
                i = gold["posx"]
                j = gold["posy"]
                if i >= my_bot_x:
                    total_gold_right += gold_amount
                if i <= my_bot_x:
                    total_gold_left += gold_amount
        count_players_left = 0
        count_players_right = 0
        for player in self.state.players:
            if "energy" in player:
                if player["status"] == self.state.STATUS_PLAYING:
                    if player["posx"] >= my_bot_x:
                        count_players_right += 1
                    if player["posx"] <= my_bot_x:
                        count_players_left += 1
            elif initial_flag:  # 0 step, initial state
                if player["posx"] >= my_bot_x:
                    count_players_right += 1
                if player["posx"] <= my_bot_x:
                    count_players_left += 1
        total_gold_left = total_gold_left / count_players_left
        total_gold_right = total_gold_right / count_players_right
        # 1 ==> left;    2 ==> both;    3 ==> right
        if total_gold_left > total_gold_right:
            return 1
        elif total_gold_left == total_gold_right:
            return 2
        else:
            return 3

    def myGetGoldAmount(self, x, y, initial_flag=False, are_we_here=False):
        distance = abs(x - self.state.x) + abs(y - self.state.y)
        gold_on_ground = self.state.mapInfo.gold_amount(x, y)
        if gold_on_ground == 0:
            return 0

        count_players = 0
        for player in self.state.players:
            if player["posx"] == x and player["posy"] == y:
                if "energy" in player:
                    if player["status"] == self.state.STATUS_PLAYING:
                        count_players += 1
                elif initial_flag:  # 0 step, initial state
                    count_players += 1
        if are_we_here:
            return gold_on_ground / count_players
        else:
            return gold_on_ground / (count_players + 1) - (distance * 50) - (50 * count_players * distance)  # +1 because assuming that we will come here

    def findLargestGold(self, initial_flag=False, leftOrRight=2):
        my_bot_x, my_bot_y = self.state.x, self.state.y
        largest_gold_x = -1
        largest_gold_y = -1

        max_gold = -100000
        for goal in self.state.mapInfo.golds:
            if leftOrRight == 2:
                if goal["amount"] > 0:
                    i = goal["posx"]
                    j = goal["posy"]

                    distance = abs(i - self.state.x) + abs(j - self.state.y)

                    count_players = 0
                    for player in self.state.players:
                        if player["posx"] == i and player["posy"] == j:
                            if "energy" in player:
                                if player["status"] == self.state.STATUS_PLAYING:
                                    count_players += 1
                            elif initial_flag:  # 0 step, initial state
                                count_players += 1

                    gold_amount = (goal["amount"] / (count_players + 1)) - (distance * 50) - (50 * count_players * distance)

                    if gold_amount > max_gold:
                        largest_gold_x = i
                        largest_gold_y = j
                        max_gold = gold_amount
                    elif gold_amount == max_gold:
                        prev_distance = (largest_gold_x - my_bot_x) * (largest_gold_x - my_bot_x) + \
                                        (largest_gold_y - my_bot_y) * (largest_gold_y - my_bot_y)
                        new_distance = (i - my_bot_x) * (i - my_bot_x) + (j - my_bot_y) * (j - my_bot_y)
                        if new_distance < prev_distance:
                            largest_gold_x = i
                            largest_gold_y = j
                            max_gold = gold_amount

            # only search at left side
            if leftOrRight == 1:
                if goal["amount"] > 0:
                    i = goal["posx"]
                    j = goal["posy"]
                    if i <= my_bot_x:
                        distance = abs(i - self.state.x) + abs(j - self.state.y)

                        count_players = 0
                        for player in self.state.players:
                            if player["posx"] == i and player["posy"] == j:
                                if "energy" in player:
                                    if player["status"] == self.state.STATUS_PLAYING:
                                        count_players += 1
                                elif initial_flag:  # 0 step, initial state
                                    count_players += 1

                        gold_amount = (goal["amount"] / (count_players + 1)) - (distance * 50) - (
                                    50 * count_players * distance)

                        if gold_amount > max_gold:
                            largest_gold_x = i
                            largest_gold_y = j
                            max_gold = gold_amount
                        elif gold_amount == max_gold:
                            prev_distance = (largest_gold_x - my_bot_x) * (largest_gold_x - my_bot_x) + \
                                            (largest_gold_y - my_bot_y) * (largest_gold_y - my_bot_y)
                            new_distance = (i - my_bot_x) * (i - my_bot_x) + (j - my_bot_y) * (j - my_bot_y)
                            if new_distance < prev_distance:
                                largest_gold_x = i
                                largest_gold_y = j
                                max_gold = gold_amount

            # only search at right side
            if leftOrRight == 3:
                if goal["amount"] > 0:
                    i = goal["posx"]
                    j = goal["posy"]
                    if i >= my_bot_x:
                        distance = abs(i - self.state.x) + abs(j - self.state.y)

                        count_players = 0
                        for player in self.state.players:
                            if player["posx"] == i and player["posy"] == j:
                                if "energy" in player:
                                    if player["status"] == self.state.STATUS_PLAYING:
                                        count_players += 1
                                elif initial_flag:  # 0 step, initial state
                                    count_players += 1

                        gold_amount = (goal["amount"] / (count_players + 1)) - (distance * 50) - (
                                50 * count_players * distance)

                        if gold_amount > max_gold:
                            largest_gold_x = i
                            largest_gold_y = j
                            max_gold = gold_amount
                        elif gold_amount == max_gold:
                            prev_distance = (largest_gold_x - my_bot_x) * (largest_gold_x - my_bot_x) + \
                                            (largest_gold_y - my_bot_y) * (largest_gold_y - my_bot_y)
                            new_distance = (i - my_bot_x) * (i - my_bot_x) + (j - my_bot_y) * (j - my_bot_y)
                            if new_distance < prev_distance:
                                largest_gold_x = i
                                largest_gold_y = j
                                max_gold = gold_amount

        return largest_gold_x, largest_gold_y

    def findLargestGoldInSmallMap(self, des_x, des_y):
        x, y = self.state.x, self.state.y
        largest_gold_x = None
        largest_gold_y = None
        next_step_x = 0
        next_step_y = 0
        if x < des_x:
            next_step_x = 1
        else:
            next_step_x = -1
        if y < des_y:
            next_step_y = 1
        else:
            next_step_y = -1

        max_gold = -100000
        while x != des_x + next_step_x:
            while y != des_y + next_step_y:
                if x != des_x or y != des_y:
                    gold_amount = self.myGetGoldAmount(x, y)
                    if gold_amount > 0:
                        if gold_amount > max_gold:
                            largest_gold_x = x
                            largest_gold_y = y
                            max_gold = gold_amount
                        elif gold_amount == max_gold:
                            prev_distance = (largest_gold_x - self.state.x) * (largest_gold_x - self.state.x) + \
                                            (largest_gold_y - self.state.y) * (largest_gold_y - self.state.y)
                            new_distance = (x - self.state.x) * (x - self.state.x) + (y - self.state.y) * (y - self.state.y)
                            if new_distance < prev_distance:
                                largest_gold_x = x
                                largest_gold_y = y
                                max_gold = gold_amount
                y += next_step_y
            y = self.state.y
            x += next_step_x

        return largest_gold_x, largest_gold_y

    def getActionBaseOnEnergy(self, action_option_1, action_option_2):
        my_bot_x, my_bot_y = self.state.x, self.state.y
        n_action = action_option_1
        require_energy = 100

        if action_option_1 == self.ACTION_GO_RIGHT:
            next_x = my_bot_x + 1
        else:
            next_x = my_bot_x - 1

        if action_option_2 == self.ACTION_GO_DOWN:
            next_y = my_bot_y + 1
        else:
            next_y = my_bot_y - 1

        energy_1 = 1
        energy_2 = 1
        gold = self.state.mapInfo.gold_amount(next_x, my_bot_y)
        if gold > 0:
            energy_1 = 4
        gold = self.state.mapInfo.gold_amount(my_bot_x, next_y)
        if gold > 0:
            energy_2 = 4

        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if i == next_x and j == my_bot_y:
                if obstacle["type"] == 1:  # Tree
                    energy_1 = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        energy_1 = 10
                elif obstacle["type"] == 3:  # Swamp
                    energy_1 = -obstacle["value"]
            if i == my_bot_x and j == next_y:
                if obstacle["type"] == 1:  # Tree
                    energy_2 = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        energy_2 = 10
                elif obstacle["type"] == 3:  # Swamp
                    energy_2 = -obstacle["value"]

        if energy_1 < energy_2:
            n_action = action_option_1
            require_energy = energy_1
        else:
            n_action = action_option_2
            require_energy = energy_2

        if self.state.energy <= require_energy:
            n_action = self.ACTION_FREE

        #print("require_energy = {0}".format(require_energy))
        #print("choose action = {0}".format(n_action))

        return n_action

    def goToTarget(self, des_x, des_y):
        n_action = self.ACTION_FREE
        require_energy = 100
        my_bot_x, my_bot_y = self.state.x, self.state.y
        next_my_bot_x = my_bot_x
        next_my_bot_y = my_bot_y
        if my_bot_x == des_x:
            if my_bot_y < des_y:
                n_action = self.ACTION_GO_DOWN
                next_my_bot_y += 1
            else:
                n_action = self.ACTION_GO_UP
                next_my_bot_y -= 1
        elif my_bot_y == des_y:
            if my_bot_x < des_x:
                n_action = self.ACTION_GO_RIGHT
                next_my_bot_x += 1
            else:
                n_action = self.ACTION_GO_LEFT
                next_my_bot_x -= 1
        else:
            if my_bot_x < des_x:
                action_option_1 = self.ACTION_GO_RIGHT
            else:
                action_option_1 = self.ACTION_GO_LEFT

            if my_bot_y < des_y:
                action_option_2 = self.ACTION_GO_DOWN
            else:
                action_option_2 = self.ACTION_GO_UP

            n_action = self.getActionBaseOnEnergy(action_option_1, action_option_2)
            return n_action

        require_energy = 1
        gold_amount = self.state.mapInfo.gold_amount(next_my_bot_x, next_my_bot_y)
        if gold_amount > 0:
            require_energy = 4
        for obstacle in self.state.mapInfo.obstacles:
            i = obstacle["posx"]
            j = obstacle["posy"]
            if i == next_my_bot_x and j == next_my_bot_y:
                if obstacle["type"] == 1:  # Tree
                    require_energy = 20
                elif obstacle["type"] == 2:  # Trap
                    if obstacle["value"] == -10:
                        require_energy = 10
                elif obstacle["type"] == 3:  # Swamp
                    require_energy = -obstacle["value"]
        if self.state.energy <= require_energy:
            n_action = self.ACTION_FREE

        return n_action

    def next_action(self, initial_flag=False):
        my_bot_x, my_bot_y = self.state.x, self.state.y
        n_action = self.ACTION_FREE
        gold_on_ground = self.myGetGoldAmount(my_bot_x, my_bot_y, initial_flag, are_we_here=True)
        energy = self.state.energy

        if gold_on_ground > 0:
            if energy <= 5:
                n_action = self.ACTION_FREE
            elif energy >= (gold_on_ground/50)*5:
                n_action = self.ACTION_CRAFT
            elif self.pre_action == self.ACTION_FREE and energy < 38:
                n_action = self.ACTION_FREE
            else:
                n_action = self.ACTION_CRAFT
        else:
            leftOrRight = 2
            if self.steps < 30:
                leftOrRight = self.goLeftOrRight(my_bot_x, my_bot_y, initial_flag)
            largest_gold_x, largest_gold_y = self.findLargestGold(initial_flag, leftOrRight)
            target_x = largest_gold_x
            target_y = largest_gold_y
            while True:
                tmp_x, tmp_y = self.findLargestGoldInSmallMap(target_x, target_y)
                if (tmp_x is None) or (tmp_y is None):
                    break
                target_x = tmp_x
                target_y = tmp_y

            n_action = self.goToTarget(target_x, target_y)

        self.steps += 1
        self.pre_action = n_action
        return n_action


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

status_map = {0: "STATUS_PLAYING", 1: "STATUS_ELIMINATED_WENT_OUT_MAP", 2: "STATUS_ELIMINATED_OUT_OF_ENERGY",
                  3: "STATUS_ELIMINATED_INVALID_ACTION", 4: "STATUS_STOP_EMPTY_GOLD", 5: "STATUS_STOP_END_STEP"}
try:
    # Initialize environment
    my_bot = MyBot(HOST, PORT)
    my_bot.start()  # Connect to the game
    my_bot.reset()
    initial_flag = True
    while not my_bot.check_terminate():
        try:
            action = my_bot.next_action(initial_flag)
            if initial_flag:
                initial_flag = False
            #print("next action = ", action)
            my_bot.step(str(action))  # Performing the action in order to obtain the new state

        except Exception as e:
            import traceback
            traceback.print_exc()
            #print("Finished.")
            break
    print(status_map[my_bot.state.status])
except Exception as e:
    import traceback
    traceback.print_exc()
#print("End game.")
