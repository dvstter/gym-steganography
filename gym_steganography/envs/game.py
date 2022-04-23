#-*- coding: utf-8 -*-
# Notice: This file is designed to self-container, so that this could be put in the gym.envs folder.
import copy

import gym
import gym_steganography.envs.utils as utils
import random
import numpy as np

class SteganographyEnv(gym.Env):
  def __init__(self, finish_points, model, model_path, folder):
    super(SteganographyEnv, self).__init__()
    self.action_space = gym.spaces.MultiDiscrete([200, 450])
    self.observation_space = gym.spaces.MultiDiscrete([200, 450, 1])

    self._finish_points = finish_points
    self._folder = folder
    self._original_state = self._load_file()
    self._state = copy.deepcopy(self._original_state)
    self._device = utils.auto_select_device()
    self._model = utils.load_model(model, model_path, self._device)

  def _load_file(self, file_num=None):
    files = utils.get_files_list(self._folder)
    file = files[file_num] if file_num else files[random.randint(0, len(files))]
    data = utils.text_read(file)
    return data[:200, :450, 0]

  def step(self, action):
    action = tuple(action)
    if abs(self._state[action]) <= 2:
      self._state[action] = -self._state[action] # flip sign bits
    done = (self._state != self._original_state).reshape(-1).sum() >= self._finish_points
    reward = 0
    if done:
      reward = self._get_reward(self._state)

    return np.expand_dims(copy.deepcopy(self._state), -1), np.array(reward, dtype=np.float), np.array(done, dtype=np.bool), {}

  def _get_reward(self, state):
    tensor_state = utils.transform(np.expand_dims(np.expand_dims(state, 0), -1), self._device)
    cover_prob = self._model.get_probabilities(tensor_state, [0])

    # reward shaping here!
    reward = cover_prob - 0.5
    return reward

  def reset(self):
    self._original_state = self._load_file()
    self._state = copy.deepcopy(self._original_state)
    # to avoid rewriting the state directly
    return np.expand_dims(copy.deepcopy(self._state), -1)