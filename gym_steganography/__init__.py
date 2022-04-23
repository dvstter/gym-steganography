from gym.envs.registration import register

register(
  id='Steganography-v0',
  entry_point='gym_steganography.envs:SteganographyEnv',
  reward_threshold=1.0,
  max_episode_steps=5000,
)