#-*- coding: utf-8 -*-
# Notice: This file is designed to self-container, so that this could be put in the gym.envs folder.
import copy
import gym
import gym_steganography.envs.utils as utils
import random
import numpy as np
import math
from collections import deque


class SteganographyEnv(gym.Env):
  def __init__(self, action_type, finish_points, model, model_path, folder, num_last_actions_usable=5):
    super(SteganographyEnv, self).__init__()
    self.observation_space = gym.spaces.Box(-10000, 10000, (200, 450), int)
    if action_type == 'discrete':
      self.action_space = gym.spaces.MultiDiscrete([200, 450])
    elif action_type == 'continuous':
      self.action_space = gym.spaces.Box(-1.0, 1.0, [2])

    self._action_type = action_type
    self._finish_points = finish_points
    self._folder = folder
    self._device = utils.auto_select_device()
    self._file_num = None
    self._last_actions_usable = deque(maxlen=100)
    self._num_last_actions_usable = num_last_actions_usable

    self._original_state, self._state = None, None
    self.reset(options={'file_num': self._file_num})
    self._model = utils.load_model(model, model_path, self._device)

    print(f'INFO:\n\tsteganography environment installed okay.\n'
          f'PARAMETERS:\n\tfinish points {finish_points}\n\t'
          f'action type: {action_type}\n\t'
          f'file folder: {folder}\n\tdevice: {self._device}\n\t'
          f'shape of states: {self._state.shape}')

  def _load_file(self, file_num=None):
    files = utils.get_files_list(self._folder)
    if file_num is None:
      file_num = random.randint(0, len(files))
    file = files[file_num]
    self._file_num = file_num
    data = utils.text_read(file)
    return data[:200, :450, 0]

  def _get_info(self):
    lau = np.array(list(self._last_actions_usable)[:self._num_last_actions_usable])
    return {'last_actions_usable': lau}

  def get_file_num(self):
    return self._file_num

  @staticmethod
  def transform_action(action, act_type=None):
    if isinstance(action[0], int):
      action = tuple([action[0]/100-1.0, action[1]/225-1.0])
    elif isinstance(action[0], float):
      action = tuple([round((action[0]+1.0)*100), round((action[1]+1.0)*225)])
    else:
      
    return action

  def _act_to_loc(self, action):
    if self._action_type == 'discrete':
      action = tuple(action)
    elif self._action_type == 'continuous':
      action = self.__class__.transform_action(action)
    return action

  def step(self, action):
    action = self._act_to_loc(action)
    if action[0] < 0 or action[0] >= 200 or action[1] < 0 or action[1] >= 450:
      self._last_actions_usable.appendleft(0)
      return copy.deepcopy(self._state), 0, False, self._get_info()
    if abs(self._state[action]) <= 2 and self._state[action] != 0:
      self._state[action] = -self._state[action] # flip sign bits
      reward = 1.0 / self._finish_points / 10 # award usable action
      self._last_actions_usable.appendleft(1)
    else:
      reward = -1.0 / self._finish_points / 10 # penalize unusable action
      self._last_actions_usable.appendleft(0)
    done = (self._state != self._original_state).reshape(-1).sum() >= self._finish_points
    if done:
      reward += self._get_reward(self._state)

#    always repeated actions, exit this project
#    if list(self._last_actions_usable)[:20] == [0]*20:
#      done = True
#      reward = -1

    return copy.deepcopy(self._state), reward, bool(done), self._get_info()

  def _get_reward(self, state):
    tensor_state = utils.transform(np.expand_dims(np.expand_dims(state, 0), -1), self._device)
    cover_prob = self._model.get_probabilities(tensor_state, [0])

    # reward reshaping here!
    reward = (cover_prob - 0.5) * 10
    return reward

  def reset(self, *, seed=None, return_info=None, options=None):
    file_num = options.get('file_num') if options else None
    try:
      self._original_state = self._load_file(file_num)
    except Exception as e:
      print(f'reset error with seed={seed} return_info={return_info} options={options} file_num={file_num}')
      exit(1)
    self._state = copy.deepcopy(self._original_state)
    # clear last N actions usable tag
    self._last_actions_usable.clear()
    for _ in range(self._num_last_actions_usable):
      self._last_actions_usable.appendleft(1)
    return copy.deepcopy(self._state)
