import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name='CartPole-v0'
env=gym.make(environment_name)

#environment_name

episodes=5
for episode in range(1,episodes+1):
    state=env.reset()
    done = False
    score=0

    while not done:
        env.render()
        action= env.action_space.sample()
        n_state, reward, done, info=env.step(action)
        score+=reward
    print('episode:{} Score:{}'.format(episode,score))
env.close()

#env.reset()

'''episodes=5
for episode in range(1,episodes+1):
    print(episode)

#env.reset()
env.step(1)'''

env.action_space.sample()
env.observation_space.sample()

env=gym.make(environment_name)
env=DummyVecEnv([lambda: env])
model=PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)

PPO_path=os.path.join ('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)
del model
model=PPO.load('PPO_model', env=env)
    
from stable_baselines3.common.evaluation import evaluate_policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

obs=env.reset()
while True:
    action, _states=model.predict(obs)
    obs, rewards, done, info=env.step(action)
    env.render()
    if done:
        print('info', info)
        break
env.close()

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
save_path= os.path.join('Training', 'Saved Models')
log_path=os.path.join('Training', 'Logs')
env=gym.make(environment_name)
env=DummyVecEnv([lambda: env])
stop_callback=StopTrainingOnRewardThreshold(reward_threshold=190, verbose=1)
eval_callback=(env, callback_on_new_best=stop_callback, eval_freq=10000, best_model_save_path=save_path, verbose=1)
model=PPO('MlpPolicy', env, verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)
model_path=os.path.join('Training', 'Saved Models', 'best_model')
model=PPO.load(model_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

net_arch=[dict(pi=[128,128,128,128], vf=[128,128,128,128])]
model=PPO('MlpPolicy',env,verbose=1, policy_kwargs={'net_arch': net_arch})
model.learn(total_timesteps=20000, callback=eval_callback)

from stable_baselines3 import DQN
model= DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=20000, callback=eval_callback)
dqn_path=os.path.join('Training', 'Saved Models', 'DQN_model')
model.save(dqn_path)
model= DQN.load(dqn_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)
env.close()

