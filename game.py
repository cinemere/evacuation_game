import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from math import sqrt

pygame.init()
font = pygame.font.Font('service_files/arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLUE3 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20  # <- USE ONLY FOR SLOW DISPLAY RUN (to use remove further commentation)

VISION_RADIUS = 30


class GameAI:

    def __init__(self, w=640, h=480):  # w=640, h=480
        self.w = w
        self.h = h
        self.vision_radius = VISION_RADIUS

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()

        self.position = None
        self.pedestrian = None
        self.pedestrian_direction = None

        self.agent_direction = None
        self.was_connected = None
        self.move_or_wait = 0  # 1 move, 0 wait
        self.score = 0
        self.frame_iteration = 0
        self.reward = 0
        self.sum_distance = 0
        self.sum_distance0 = 0
        self.reset()

    def reset(self, games_recorded=4):
        # init game state
        self.move_or_wait = 1  # 1 move, 0 wait
        self.score = 0
        self.frame_iteration = 0
        self.reward = 0

        self.agent_direction = Direction.RIGHT

        if games_recorded == 30 or games_recorded == 35:
            self._place_simple()  # self._place_food()
        else:
            self._place_pedestrian()

        self.was_connected = self.leader_in_vision()
        self.sum_distance = self._sum_distance()
        self.sum_distance0 = self.sum_distance

    def _place_simple(self):
        self.position = Point(self.w / 2, self.h / 2)
        self.exit = Point(self.w / 2 - BLOCK_SIZE, self.h / 2)
        self.pedestrian = Point(self.w / 2, self.h / 2 - BLOCK_SIZE)
        self.pedestrian_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

    def _place_pedestrian(self):
        self.position = Point(self.w / 2, self.h / 2)
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.exit = Point(x, y)
        # self.exit = Point(self.w - BLOCK_SIZE, self.h / 2)  # Point(x, y)
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.pedestrian = Point(x, y)
        self.pedestrian_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])
        if self.pedestrian == self.exit:  # in
            self._place_pedestrian()

    def _sum_distance(self):
        # d = ((self.pedestrian.x - self.exit.x) ** 2 + (self.pedestrian.y - self.exit.y) ** 2) ** 0.5
        # d += ((self.pedestrian.x - self.position.x) ** 2 + (self.pedestrian.y - self.position.y) ** 2) ** 0.5
        d = abs(self.pedestrian.x - self.exit.x) / BLOCK_SIZE
        d += abs(self.pedestrian.y - self.exit.y) / BLOCK_SIZE
        d += abs(self.pedestrian.x - self.position.x) / BLOCK_SIZE * 2
        d += abs(self.pedestrian.y - self.position.y) / BLOCK_SIZE * 2
        return d

    def play_step(self, action):
        self.reward = 0
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        # self.snake.insert(0, self.head)

        # 4. return some reward
        self.score -= 1
        new_sum_dist = self._sum_distance()
        # self.reward = - new_sum_dist
        self.reward = self.sum_distance - new_sum_dist
        self.sum_distance = new_sum_dist

        # 3. check if game over
        game_over = False
        if self.frame_iteration > self.h/5 + self.w/5 \
                or self.is_collision():  # 300:  #* len(self.snake):  # self.is_collision() or
            game_over = True
            self.score = -1000
            self.reward = -self._sum_distance()
            return self.reward, game_over, self.score

        if self.pedestrian == self.exit:
            game_over = True
            self.score = - self.frame_iteration
            self.reward = 1000 - self.frame_iteration
            # self._place_pedestrian()

        # 5. update ui and clock
        self._update_ui()
        # self.clock.tick(SPEED)  # <-- for AI FAST RUN PUT COMMENT
        # 6. return game over and score
        return self.reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.position
        # if pt == self.exit:
        #    return False
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or \
                pt.x < 0 or \
                pt.y > self.h - BLOCK_SIZE or \
                pt.y < 0:
            return True
        # hits itself
        # if pt in self.snake[1:]:
        #     return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # for pt in self.snake:
        #     pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        #     pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pt = self.position  # leader agent  TODO: add direction, move/wait
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pt = self.pedestrian  # pedestrian
        pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        pt = self.exit  # exit
        pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Step: " + str(self.frame_iteration)
                           + "   Dist: " + str(int(self.sum_distance))
                           + "   Reward:" + str(self.reward)
                           + "   " + str(self.agent_direction), True,
                           WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def leader_in_vision(self):
        # distance = sqrt((self.position.x - self.pedestrian.x) ** 2 + (self.position.y - self.pedestrian.y) ** 2)
        # if distance < VISION_RADIUS:
        if abs(self.position.x - self.pedestrian.x) < VISION_RADIUS \
                and abs(self.position.y - self.pedestrian.y) < VISION_RADIUS:
            return True
            # if self.pedestrian_direction == Direction.LEFT:
            #     direction_vector = np.array([-1, 0])
            # elif self.pedestrian_direction == Direction.RIGHT:
            #     direction_vector = np.array([1, 0])
            # elif self.pedestrian_direction == Direction.UP:
            #     direction_vector = np.array([0, -1])
            # else:  # DOWN
            #     direction_vector = np.array([0, 1])
            #
            # destination_vector = np.array([self.position.x - self.pedestrian.x,
            #                                self.position.y - self.pedestrian.y])
            #
            # vector_mult = np.dot(direction_vector, destination_vector)
            #
            # if vector_mult > 0:
            #     return True
        return False

    def _move(self, action):
        # [straigh, right, left, backward]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.agent_direction)


        action_direction = action  # [0:-1]
        # print(action)
        action_move_or_wait = 1  # action[-1]

        if np.array_equal(action_direction, [1, 0, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action_direction, [0, 1, 0, 0]):
            next_idx = (idx + 1) % 4  # right turn r -> d -> l -> u -> r
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action_direction, [0, 0, 1, 0]):  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # left turn r -> u -> l -> d -> r
            new_dir = clock_wise[next_idx]
        elif np.array_equal(action_direction, [0, 0, 0, 1]):
            next_idx = (idx + 2) % 4  # turn backwards r <-> l;  u <-> d
            new_dir = clock_wise[next_idx]

        self.agent_direction = new_dir

        if action_move_or_wait == 1:

            x = self.position.x
            y = self.position.y

            if self.agent_direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.agent_direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.agent_direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.agent_direction == Direction.UP:
                y -= BLOCK_SIZE

            new_position = Point(x, y)

            # if not self.is_collision(new_position):

            self.position = new_position

        if self.leader_in_vision():
            # self.reward += 1
            self.pedestrian_direction = self.agent_direction

            x = self.pedestrian.x
            y = self.pedestrian.y

            if self.pedestrian_direction == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.pedestrian_direction == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.pedestrian_direction == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.pedestrian_direction == Direction.UP:
                y -= BLOCK_SIZE

            new_position = Point(x, y)

            if not self.is_collision(new_position):
                self.pedestrian = new_position

        else:
            self.pedestrian_direction = random.choice([Direction.RIGHT, Direction.LEFT, Direction.UP, Direction.DOWN])

            ##
