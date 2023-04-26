from gym.envs.registration import register

register(
  id='Steganography-v0',
  entry_point='gym_steganography.envs:SteganographyEnv',
  reward_threshold=1.0,
  max_episode_steps=100000,
)

# make creating test environment easier
import gym
def make(action_type='continuous', finish_points=1000, model='rhfcn', model_path='/home/hanlinyoung/stego_analysis/adversarial/model_rhfcn_local.pth', folder='/home/hanlinyoung/stego_analysis/train_data/Cover/320/', num_last_actions_usable=5):
  return gym.make('Steganography-v0',
                  action_type=action_type,
                  finish_points=finish_points,
                  model=model,
                  model_path=model_path,
                  folder=folder,
                  num_last_actions_usable=num_last_actions_usable)
