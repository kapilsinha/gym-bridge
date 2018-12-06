import numpy as np
import time
from gym_bridge.envs.bridge_env import BridgeEnv, check_game_status,\
    after_action_state


def match(team_1, team_2):
    '''
    Plays a single match between two teams.
    
    Arguments:
        team_1 (dict) - a map from player name to an agent with an act method
                      - player names are 'West', 'North', 'East', 'South'
        team_2 (dict)
    
    Returns a boolean indicating if team 1 won.
    '''
    
    # flip a coin to decide which team gets EW
    if np.random.random() < 0.5:
        EW = team_1
        NS = team_2
        rep = 'West' # representative from team 1
    else:
        EW = team_2
        NS = team_1
        rep = 'North' # representative from team 1
        
    agents = {
        'West': EW['West'],
        'East': EW['East'],
        'North': NS['North'],
        'South': NS['South']
    }
    
    env = BridgeEnv()
    observation = env.reset(start_player_name='West')
        
    done = False
    while not done:
        done, obs, info = check_game_status(env)
        curr_agent = agents[info['cur_player']]
            
        action = curr_agent.act(obs, env)
        obs, reward, done, info = env.step(action)
        
    win_team_1 = reward[rep] > 0
    return win_team_1
    
def tournament(teams, n_games=1000):
    '''
    Given a vector of teams, simulate many games for each possible pairing of
    teams and return the results.
    
    Arguments:
        teams - a list of teams
        n_games - the number of games played between each pairing of teams.
    
    Returns a shape (N, N) array of win_rates (between 0 and 1).
    '''
    
    N = len(teams)
    win_rate = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            n_wins = sum(match(teams[i], teams[j]) for n in range(n_games))
            win_rate[i, j] = n_wins / n_games
            win_rate[j, i] = 1 - n_wins / n_games
            
    return win_rate

def pretty_print_list(A):
    
    A_str = [[str(e) for e in row] for row in A]
    width = 1 + max(len(s) for row in A_str for s in row)
    
    for row in A_str:
    
        row_str = ''
        for s in row:
            row_str += s + ' ' * (width - len(s))
            
        print(row_str)

if __name__ == '__main__':

    from betting_agent import BettingAgent
    from base_agent import BaseAgent
    from policy_gradient_agent import PassAgent, train_vs_pass, train_vs_self
    
    iterations, episodes = 250, 1000
    n_games = 1000
    
    print('Training teams...\n')
    # TODO: figure out how to save/load pytorch weights
    players = ['West', 'North', 'East', 'South']
    team_map = {
        'PassAgent': {name: PassAgent() for name in players},
        'BaseAgent': {name: BaseAgent(name) for name in players},
        # 'BettingAgent': {name: BettingAgent(name) for name in players}, # currently broken
        'PGAgent1': train_vs_pass(iterations, episodes),
        'PGAgent2': train_vs_self(iterations, episodes),
    }
    n_teams = len(team_map)
    names = list(team_map.keys())
    teams = list(team_map.values())
    
    print('Running tournament...\n')
    start = time.time()
    results = tournament(teams, n_games=n_games)
    end = time.time()
    print('Tournament concluded, taking %.3f s' % (end - start))
    
    results = np.round(results, 2).tolist()
    results = [[names[i]] + results[i] for i in range(n_teams)]
    results = [[''] + names] + results
    print("Results (win-rate of row team over %d games)\n" % n_games)
    pretty_print_list(results)