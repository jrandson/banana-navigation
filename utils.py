import numpy as np

def create_uniform_grid(low, high, bins=10):
    discrete_values = np.linspace(low, high, bins+1)[1:-1]  
    return discrete_values


def discretize(sample, grid):
    #return tuple([np.digitize(sample[i], grid[i]) for i in range(len(grid))])
    return np.digitize(sample, grid)
    #return list(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def run(agent, env, num_episodes=20000, mode='train'):
    """Run agent in given reinforcement learning environment and return scores."""
    scores = []
    max_avg_score = -np.inf
    for i_episode in range(1, num_episodes+1):
        # Initialize episode
        state = agent.state_grid[0]
        action = agent.reset_episode(state)
		
        if mode == 'train':
            env_info = env.reset(train_mode=True)[brain_name]
			
        total_reward = 0
        done = False

        # Roll out steps until done
        while not done:
            sys.stdout.flush()
            state, reward, done = agent.step(action)
            total_reward += reward
            action = agent.act(state, reward, done, mode)  
        
        # Save final score
        scores.append(total_reward)
        
        # Print episode stats
        if mode == 'train':
            if len(scores) > 100:
                avg_score = np.mean(scores[-100:])
                if avg_score > max_avg_score:
                    max_avg_score = avg_score

            if i_episode % 100 == 0:
                print("\rEpisode {}/{} | Max Average Score: {}".format(i_episode, num_episodes, max_avg_score), end="")
                sys.stdout.flush()

    return scores