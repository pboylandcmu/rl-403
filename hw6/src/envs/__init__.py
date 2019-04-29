from gym.envs.registration import register

register(
    id='Pushing2D-v1',
    entry_point='envs.2Dpusher_env:Pusher2d'
)
