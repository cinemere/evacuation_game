import torch
import random
import numpy as np
from collections import deque
from game import GameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

BLOCK_SIZE = 20

INPUT_SIZE = 21  # 11
HIDDEN_SIZE = 512
OUTPUT_SIZE = 4


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.8  # discount < 1
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        agent = game.position
        exit = game.exit

        point_l = Point(agent.x - BLOCK_SIZE, agent.y)
        point_r = Point(agent.x + BLOCK_SIZE, agent.y)
        point_u = Point(agent.x, agent.y - BLOCK_SIZE)
        point_d = Point(agent.x, agent.y + BLOCK_SIZE)

        dir_l = game.agent_direction == Direction.LEFT
        dir_r = game.agent_direction == Direction.RIGHT
        dir_u = game.agent_direction == Direction.UP
        dir_d = game.agent_direction == Direction.DOWN

        state = [
            (dir_l and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),  # wall straight

            (dir_l and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_u)),  # wall back

            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),  # wall right

            (dir_l and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),  # wall left

            dir_l,  # direction left
            dir_r,  # direction right
            dir_u,  # direction up
            dir_d,  # direction down

            game.pedestrian.x < agent.x,  # pedestrian left
            game.pedestrian.x > agent.x,  # pedestrian right
            game.pedestrian.y < agent.y,  # pedestrian up
            game.pedestrian.y > agent.y,   # pedestrian down

            game.leader_in_vision(),  # can pedestrian see agent

            abs(game.pedestrian.x - agent.x) / game.w,
            abs(game.pedestrian.y - agent.y) / game.h,
            abs(game.pedestrian.x - exit.x) / game.w,
            abs(game.pedestrian.y - exit.y) / game.h,

            exit.x < agent.x,  # exit left
            exit.x > agent.x,  # exit right
            exit.y < agent.y,  # exit up
            exit.y > agent.y   # exit down

            #agent.x / game.w,
            #agent.y / game.h
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if max memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # random sample
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):  # one step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 150 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 400) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = prediction.tolist()
            # direction
            move_idx = np.argmax(move)
            final_move[move_idx] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    plot_rewards = []
    plot_mean_rewards = []
    total_reward = 0
    record_reward = 0

    agent = Agent()
    game = GameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get next state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot
            record_reward = game.reward
            game.reset(len(plot_scores))
            agent.n_games += 1

            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # plot_rewards.append(record_reward)
            print(record_reward)
            # total_reward += record_reward
            # mean_rewards = total_reward / agent.n_games
            # plot_mean_rewards.append(mean_rewards)
            # plot(plot_rewards, plot_mean_rewards)



if __name__ == '__main__':
    train()