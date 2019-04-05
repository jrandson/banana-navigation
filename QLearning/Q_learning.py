class QLearningAgent:
    """Q-Learning agent that can act on a continuous state space by discretizing it."""

    def __init__(self, env, state_grid, brain_name, alpha=0.02, gamma=0.99,
                 epsilon=1.0, epsilon_decay_rate=0.9995, min_epsilon=.01, seed=505):
        """Initialize variables, create grid for discretization."""
        # Environment info
        self._brain_name = brain_name
        self.brain = env.brains[brain_name]
        self.env_info = env.reset(train_mode=True)[brain_name]
        self.env = env
        self.state_grid = state_grid
        self.state_size = tuple(self.env_info.vector_observations[0].shape)  # n-dimensional state space
       
        self.action_size = self.brain.vector_action_space_size   # 1-dimensional discrete action space
        self.seed = np.random.seed(seed)
        
        print("Environment:", self.env)
        print("State space size:", self.state_size)
        print("Action space size:", self.action_size)
        
        # Learning parameters
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = self.initial_epsilon = epsilon  # initial exploration rate
        self.epsilon_decay_rate = epsilon_decay_rate # how quickly should we decrease epsilon
        self.min_epsilon = min_epsilon
        
        # Create Q-table
        print(self.state_size)
        print(self.action_size)
        self.q_table = np.zeros(shape=(self.state_size + (self.action_size,)))
                                        
        print("Q table size:", self.q_table.shape)

    def preprocess_state(self, state):
        """Map a continuous state to its discretized representation."""
        # TODO: Implement this
        
        return discretize(state, self.state_grid)

    def reset_episode(self, state, mode='train'):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
        if mode == 'train':
            self.env.reset(train_mode=True)[self._brain_name]
					
        # Decide initial action
        self.last_state = self.preprocess_state(state)
        self.last_action = np.argmax(self.q_table[self.last_state])
        return self.last_action
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon

    def step(self, action):
        self.env_info = env.step(action)[self._brain_name]
        next_state = self.env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        return next_state, reward, done
        

    def act(self, state, reward=None, done=None, mode='train'):
        """Pick next action and update internal Q table (when mode != 'test')."""
        
        state = self.preprocess_state(state)[0]
        if mode == 'test':
            # Test mode: Simply produce an action
            action = np.argmax(self.q_table[state])
        else:
            
            # Train mode (default): Update Q table, pick next action
            # Note: We update the Q table entry for the *last* (state, action) pair with current state, reward#           
          
            self.q_table[self.last_state][self.last_action] += self.alpha * \
                (reward + self.gamma * max(self.q_table[state]) - self.q_table[self.last_state][self.last_action])
        
            
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.q_table[state])
                

        # Roll over current state, action for next step
        self.last_state = state
        self.last_action = action
        return action