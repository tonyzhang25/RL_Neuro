import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_agent_data(dir):
    return pickle.load(open(f"{dir}/sess_data.p", 'rb'))


agent_data = load_agent_data('/home/tzhang/Dropbox/Caltech/VisionLab/RL/Neuro/RL_Neuro/data/analysis/experiment_50/sess_0')

import pdb; pdb.set_trace()
state_action_log = agent_data.state_act_history_trials



def analyze_agent_novel_states_visited(state_action_log, truncate = 2000):
    all_agents_sameconfig = []
    for trial_data in state_action_log:
        one_agent_novel_states = []
        visited_states = set()
        for bout in trial_data: # each episode is one bout
            for state, _, _, in bout:
                visited_states.add(state)
                one_agent_novel_states.append(len(visited_states))
            # +1 at end because terminal state not counted
            one_agent_novel_states.append(one_agent_novel_states[-1])
        all_agents_sameconfig.append(one_agent_novel_states)

    if any(np.array([len(states) for states in all_agents_sameconfig]) < truncate):
        print(f'* Issue: at least one agent does not have enough data (<{truncate} nodes)')
        import pdb; pdb.set_trace()
    else:
        # trim each trial, and find variance / SD
        data_for_averaging = []
        for agent_data in all_agents_sameconfig:
            data_for_averaging.append(agent_data[:truncate])
        # compute variance
        var = np.std(data_for_averaging, axis = 0)
        avg = np.average(data_for_averaging, axis = 0)
        plt.figure()
        x = [i for i in range(truncate)]
        color = 'C0'
        plt.fill_between(x, avg - var, avg + var, facecolor = color, alpha = 0.3)
        plt.plot(avg, color = color)
        plt.savefig('/media/tzhang/Tony_WD_4TB_2/5_maze_v4/B_animals_reward/analysis/test2.png',
                    bbox_inches = 'tight', )

analyze_agent_novel_states_visited(state_action_log)