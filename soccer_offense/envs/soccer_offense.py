import gym
from gym import spaces
import pygame
import numpy as np
from copy import deepcopy


# action dim = 8
# state dim = 12


class SoccerOffenseEnv(gym.Env):
	metadata = {
		'render_modes': ['human', 'rgb_array'],
		'render_fps': 10
	}
	INIT_STRIKER_RANGE_X = (0, 1)  # limit the position where the striker could spawn at
	INIT_STRIKER_RANGE_Y = (0, 0.1)
	GOALIE_POS_RANGE_X = (0.4, 0.6)  # limit the goalies spawn position
	GOALIE_POS_RANGE_Y = (0.9, 1)
	INIT_BALL_RANGE_X = (0.2, 0.8)  # limit the goalies spawn position
	INIT_BALL_RANGE_Y = (0.1, 0.2)
	GOAL_RANGE_X = (0.25, 0.75)  # goal is always on y = 1
	GOAL_LEFT_RANGE = (GOAL_RANGE_X[0] - 0.05, GOAL_RANGE_X[0] + 0.15)  # actual shoot direction sampled from ranges
	GOAL_MID_RANGE = (GOAL_RANGE_X[0] + 0.1, GOAL_RANGE_X[1] - 0.1)
	GOAL_RIGHT_RANGE = (GOAL_RANGE_X[1] - 0.15, GOAL_RANGE_X[1] + 0.05)

	FRICTION_BALL = 0.0016  # scalar friction
	FRICTION_BALL_FACTOR = 0.08  # speed-relative friction = spd * friction_factor
	FRICTION_PLAYER = 0.005  # must be smaller than STRIKER_ACC

	STRIKER_ACC = 0.01  # acceleration
	STRIKER_TOP_SPD = 0.04  # top speed
	GOALIE_ACC = 0.01
	GOALIE_TOP_SPD = 0.03
	SHOOT_BALL_SPD = 0.1

	STRIKER_CARRY_RANGE = 0.05  # radius determines if the player can reach the ball
	GOALIE_CATCH_RANGE = 0.05

	def __init__(self,
	             render_mode=None,
	             goalie_mode='chase'):
		"""
			A striker is getting to get the ball into the goal while a goalie is trying to
		prevent it from happen.
			At start of the game, the striker and the ball randomly spawn at one side of
		the filed while the goalie spawns randomly at the opposite side near the goal.
			If the ball is in certain range around striker, striker will carry the ball automatically
		while moving and be able to shoot. the striker can choose to shoot at either left, middle
		or right area of the goal, there is a small chance of missing shooting left or right, while
		shooting at middle will not miss.
			The ball will be captured by goalie once the ball is close enough to the goalie,
		and the game is over once captured. Goalie is controlled by environment with goalie_mode deciding
		the behavior of the goalie.
			The game is over once:
				1. the striker runs outside the field (bad ending)
				2. the ball goes outside the field (bad ending)
				3. the ball is captured by goalie (bad ending)
				4. the striker achieves a goal (good ending)
			The agent/human player plays the striker, the first reasonable thing to learn/do is to move
		toward the ball and reach it, after that, the striker shall find his/her best chance to shoot
		and achieve a goal.
			The friction slows the ball down, makes it possible that the ball stops before the goal,
		and may forcing the striker to learn how to bypass the goalie to get closer to the goal.
			The original game is extremely reward-sparse: only get one reward at the end of the game. you
		can modify the reward under certain circumstances. See self._apply_rules for reward details.
		:param render_mode: None, 'human' or 'rgb_array'
		:param goalie_mode: could be 'stay' or 'chase', default is 'chase'
		"""

		self.goalie_mode = goalie_mode
		# [striker_pos, striker_spd, ball_pos, ball_spd, goalie_pos, goalie_spd]
		self.observation_space = spaces.Box(low=-0.1, high=1.1, shape=(12,))
		# "none", "up", "down", "left", "right", "shoot-left", "shoot-middle", "shoot-right"
		self.action_space = spaces.Discrete(8)

		# state-determine attributes
		self.striker_pos = None  # [0, 1]
		self.striker_spd = None  # [0, 1] -> real speed=[0, 1 * GOALIE_TOP_SPD]
		self.goalie_pos = None
		self.goalie_spd = None  # [0, 1] -> real speed=[0, 1 * GOALIE_TOP_SPD]
		self.ball_pos = None
		self.ball_spd = None  # [0, 1] -> real speed=[0, 1 * SHOOT_BALL_SPD]
		self.ball_state = 'free'
		self.flag_striker_first_time_carry = False
		self.flag_striker_first_time_shoot = False
		self.step_ctr = 0

		# actions to accelerations
		self.action2acc_striker = {
			0: np.array([0, 0]),
			1: np.array([0, self.STRIKER_ACC]),
			2: np.array([0, -self.STRIKER_ACC]),
			3: np.array([-self.STRIKER_ACC, 0]),
			4: np.array([self.STRIKER_ACC, 0]),
			5: np.array([0, 0]),  # shoot
			6: np.array([0, 0]),
			7: np.array([0, 0])
		}
		self.action2acc_goalie = {
			0: np.array([0, 0]),
			1: np.array([0, self.GOALIE_ACC]),
			2: np.array([0, -self.GOALIE_ACC]),
			3: np.array([-self.GOALIE_ACC, 0]),
			4: np.array([self.GOALIE_ACC, 0]),
			5: np.array([0, 0]),  # shoot
			6: np.array([0, 0]),
			7: np.array([0, 0])
		}

		self.reset()

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode
		self.window_size = 512
		self.window = None
		self.clock = None

	def get_state_dict(self):
		"""get the dict recording all necessary attribute determine the state"""
		return {
			'striker_pos': deepcopy(self.striker_pos),
			'striker_spd': deepcopy(self.striker_spd),
			'goalie_pos': deepcopy(self.goalie_pos),
			'goalie_spd': deepcopy(self.goalie_spd),
			'ball_pos': deepcopy(self.ball_pos),
			'ball_spd': deepcopy(self.ball_spd),
			'ball_state': deepcopy(self.ball_state),
			'flag_carry': deepcopy(self.flag_striker_first_time_carry),
			'flag_shoot': deepcopy(self.flag_striker_first_time_shoot),
			'step_ctr': deepcopy(self.step_ctr)
		}

	def load_state_dict(self, state_dict):
		"""load state dict to set the game at a certain state"""
		self.striker_pos = deepcopy(state_dict['striker_pos'])
		self.striker_spd = deepcopy(state_dict['striker_spd'])
		self.goalie_pos = deepcopy(state_dict['goalie_pos'])
		self.goalie_spd = deepcopy(state_dict['goalie_spd'])
		self.ball_pos = deepcopy(state_dict['ball_pos'])
		self.ball_spd = deepcopy(state_dict['ball_spd'])
		self.ball_state = deepcopy(state_dict['ball_state'])
		self.flag_striker_first_time_carry = deepcopy(state_dict['flag_carry'])
		self.flag_striker_first_time_shoot = deepcopy(state_dict['flag_shoot'])
		self.step_ctr = deepcopy(state_dict['step_ctr'])

	def _goalie_agent(self):
		"""goalie behavior mode, can have more behavior mode"""
		if self.goalie_mode == 'stay':  # do nothing
			action = 0

		elif self.goalie_mode == 'chase':
			if self.ball_state == 'caught':
				action = 0
			else:
				ball_direction = self.ball_pos - self.goalie_pos
				ball_direction /= np.linalg.norm(ball_direction)

				vertical = 1 if ball_direction[1] > 0 else 2
				horizontal = 3 if ball_direction[0] < 0 else 4
				# since only one action can be chosen at a time,
				# randomly sample one action that helps the goalie gets to the ball
				if np.random.uniform(0, np.sum(np.abs(ball_direction))) < np.abs(ball_direction[0]):
					action = horizontal
				else:
					action = vertical
		else:
			action = 0
		return action

	def _get_obs(self):
		"""get observations"""
		return np.concatenate([
			self.striker_pos,
			self.striker_spd,
			self.ball_pos,
			self.ball_spd,
			self.goalie_pos,
			self.goalie_spd,
		]).astype(np.float32)

	@staticmethod
	def _limit_spd(spd, limit):
		"""limit the speed"""
		spd_mag = np.linalg.norm(spd)
		if spd_mag > limit:
			spd *= (limit / spd_mag)
		return spd

	@staticmethod
	def _limit_pos(pos, x1, x2, y1, y2):
		"""limit the position"""
		pos[0] = np.clip(pos[0], x1, x2)
		pos[1] = np.clip(pos[1], y1, y2)
		return pos

	@staticmethod
	def _speed_decay(speed, friction, friction_factor=0.):
		"""friction effect on speed"""
		spd_mag = np.linalg.norm(speed)
		friction_tol = friction + spd_mag * friction_factor
		if spd_mag < friction_tol:
			speed *= 0
		else:
			speed -= friction_tol * speed / spd_mag
		return speed

	def _update_ball_state(self):
		if np.linalg.norm(self.goalie_pos - self.ball_pos) < self.GOALIE_CATCH_RANGE:
			self.ball_state = 'caught'
		elif np.linalg.norm(self.striker_pos - self.ball_pos) < self.STRIKER_CARRY_RANGE:
			self.ball_state = 'carried'
		else:
			self.ball_state = 'free'

	def _update_ball(self, action):
		self._update_ball_state()
		if self.ball_state == 'caught':
			self.ball_spd = self.goalie_spd
			if self.goalie_spd.any():
				self.ball_pos = self.goalie_pos \
				                + self.goalie_spd * 0.5 * self.GOALIE_CATCH_RANGE \
				                / np.linalg.norm(self.goalie_spd + 0.00001)

		elif self.ball_state == 'carried':
			if action > 4:  # shoot!
				if action == 5:  # shoot left
					aim_pts = np.random.uniform(*self.GOAL_LEFT_RANGE)
				elif action == 6:  # shoot middle
					aim_pts = np.random.uniform(*self.GOAL_MID_RANGE)
				else:  # action ==7  shoot right
					aim_pts = np.random.uniform(*self.GOAL_RIGHT_RANGE)
				aim_line = np.array([aim_pts, 1]) - self.ball_pos
				self.ball_spd = aim_line * self.SHOOT_BALL_SPD \
				                / np.linalg.norm(aim_line)
				self.ball_pos += self.ball_spd
				self.ball_state = 'free'
			else:  # keep carrying
				self.ball_spd = self.striker_spd
				if self.striker_spd.any():
					self.ball_pos = self.striker_pos \
					                + self.striker_spd * 0.7 * self.STRIKER_CARRY_RANGE \
					                / np.linalg.norm(self.striker_spd + 0.00001)
		else:  # self.ball_state == 'free'
			self.ball_spd = self._speed_decay(self.ball_spd,
			                                  self.FRICTION_BALL,
			                                  self.FRICTION_BALL_FACTOR)
			self.ball_pos += self.ball_spd

	@staticmethod
	def _is_outside_field(pos):
		"""check if the given position is outside the field"""
		return False if np.all((0 < pos) & (pos < 1)) else True

	def _is_goal(self):
		"""check if the ball is in the goal"""
		left, right = self.GOAL_RANGE_X
		return True if left < self.ball_pos[0] < right and self.ball_pos[1] >= 1 \
			else False

	def _apply_rules(self, action):
		"""
		calculate reward and termination on given state and action
		you can modify the reward under different circumstances
		"""

		term = False
		r = 0

		# achieve a goal
		if self._is_goal():
			term = True
			r += 100

		# ball out of field
		if self._is_outside_field(self.ball_pos) and not self._is_goal():
			term = True
			r += -5

		# striker out of field
		if self._is_outside_field(self.striker_pos):
			term = True
			r += -5

		# goalie captures the ball
		if self.ball_state == 'caught':
			term = True
			r += -5

		# striker carry the ball for the first time
		if self.ball_state == 'carried' and not self.flag_striker_first_time_carry:
			self.flag_striker_first_time_carry = True
			r += 0

		# striker shoot for the first time
		if action > 4 and self.ball_state == 'carried' and not self.flag_striker_first_time_shoot:
			self.flag_striker_first_time_shoot = True
			r += 0

		# striker approaching to the ball
		if self.ball_state == 'free':
			ball_dir = self.ball_pos - self.striker_pos
			cos_sim = np.dot(self.striker_spd, ball_dir) / \
			          (np.linalg.norm(self.striker_spd) * np.linalg.norm(ball_dir)) \
				if self.striker_spd.any() else 0
			# reward_factor * cosine similarity
			r += 0 * cos_sim

		return r, term

	def reset(self, seed=None, options=None):
		"""random initialize positions of striker, goalie and ball """
		super(SoccerOffenseEnv, self).reset(seed=seed)
		self.striker_pos = np.array([
			np.random.uniform(*self.INIT_STRIKER_RANGE_X),
			np.random.uniform(*self.INIT_STRIKER_RANGE_Y)
		])
		self.striker_spd = np.array([0, 0])
		self.ball_pos = np.array([
			np.random.uniform(*self.INIT_BALL_RANGE_X),
			np.random.uniform(*self.INIT_BALL_RANGE_Y)
		])
		self.ball_spd = np.array([0, 0])
		self.goalie_pos = np.array([
			np.random.uniform(*self.GOALIE_POS_RANGE_X),
			np.random.uniform(*self.GOALIE_POS_RANGE_Y)
		])
		self.goalie_spd = np.array([0, 0])
		self.flag_striker_first_time_shoot = False
		self.flag_striker_first_time_carry = False
		self.step_ctr = 0
		return self._get_obs(), {}

	def step(self, action):
		"""state transition on action"""
		assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

		# update speed => speed += (acceleration - friction)
		self.striker_spd = self._speed_decay(self.striker_spd, self.FRICTION_PLAYER) \
		                   + self.action2acc_striker[action]
		self.striker_spd = self._limit_spd(self.striker_spd, self.STRIKER_TOP_SPD)
		self.striker_pos += self.striker_spd

		self.goalie_spd = self._speed_decay(self.goalie_spd, self.FRICTION_PLAYER) \
		                  + self.action2acc_goalie[self._goalie_agent()]
		self.goalie_spd = self._limit_spd(self.goalie_spd, self.GOALIE_TOP_SPD)
		self.goalie_pos += self.goalie_spd
		self._update_ball(action)

		self.striker_pos = self._limit_pos(self.striker_pos, -0.1, 1.1, -0.1, 1.1)
		self.goalie_pos = self._limit_pos(self.goalie_pos, -0.1, 1.1, -0.1, 1.1)
		self.ball_pos = self._limit_pos(self.ball_pos, -0.1, 1.1, -0.1, 1.1)
		self.step_ctr += 1
		r, t = self._apply_rules(action)
		obs = self._get_obs()
		return obs, r, t, False, {}

	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()
		else:
			self._render_frame()

	def _render_frame(self):
		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode((self.window_size, self.window_size))
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		canvas = pygame.Surface((self.window_size, self.window_size))
		canvas.fill((50, 150, 50))

		# draw striker
		pygame.draw.circle(
			canvas,
			color=(0, 0, 255),
			center=self.striker_pos * self.window_size,
			radius=10
		)
		# draw ball
		pygame.draw.circle(
			canvas,
			color=(255, 255, 255),
			center=self.ball_pos * self.window_size,
			radius=6
		)
		# draw goalie
		pygame.draw.circle(
			canvas,
			color=(255, 0, 0),
			center=self.goalie_pos * self.window_size,
			radius=10
		)

		# draw goal
		pygame.draw.line(
			canvas,
			color=(255, 255, 255),
			start_pos=np.array([self.GOAL_RANGE_X[0], 1]) * self.window_size,
			end_pos=np.array([self.GOAL_RANGE_X[1], 1]) * self.window_size,
			width=10
		)

		if self.render_mode == "human":
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()
			self.clock.tick(self.metadata["render_fps"])
		else:  # rgb_array
			return np.transpose(
				np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
			)

	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()

	def play(self):
		"""
		human plays the game with keyboard,
		WASD for moving,
		Arrow keys for shooting: left=shoot_l, down=shoot_m, right=shoot_r
		"""
		while True:
			_, _ = self.reset()
			term = False
			while not term:
				self.render()
				keys = pygame.key.get_pressed()
				if keys[pygame.K_s]:
					action = 1
				elif keys[pygame.K_w]:
					action = 2
				elif keys[pygame.K_a]:
					action = 3
				elif keys[pygame.K_d]:
					action = 4
				elif keys[pygame.K_LEFT]:
					action = 5
				elif keys[pygame.K_DOWN]:
					action = 6
				elif keys[pygame.K_RIGHT]:
					action = 7
				else:
					action = 0
				ob, r, term, _, _ = self.step(action)
