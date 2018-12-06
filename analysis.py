import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_bridge
import time

env = gym.make('bridge-v0')

n_games = 10000
points = []
team_points = []

for g in range(n_games):
    
    env.reset()
    W = env.state.players['West'].points
    E = env.state.players['East'].points
    points.append(W)
    team_points.append(W + E)
    
print('Mean points: %.3f' % np.mean(points))
print('St.dev points: %.3f' % np.std(points))
print('Min points: %.3f' % np.min(points))
print('Max points: %.3f' % np.max(points))
print('')
print('Mean team points: %.3f' % np.mean(team_points))
print('St.dev team points: %.3f' % np.std(team_points))
print('Min team points: %.3f' % np.min(team_points))
print('Max team points: %.3f' % np.max(team_points))

plt.figure()
plt.hist(points, bins=np.arange(min(points), 1 + max(points)))
plt.title('Points')
plt.show()

plt.figure()
plt.hist(team_points, bins=np.arange(min(team_points), 1 + max(team_points)))
plt.title('Team Points')
plt.show()