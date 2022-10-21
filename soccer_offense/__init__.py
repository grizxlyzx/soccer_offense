from gym.envs.registration import register

register(
    id='soccer_offense/SoccerOffense-v0',
    entry_point='soccer_offense.envs:SoccerOffenseEnv',
    # max_episode_steps=1000,
    kwargs={'goalie_mode': 'stay'}
)