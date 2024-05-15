class utils():

    def argmax(self, rand_generator, q_values):
        """argmax with random tie-breaking
        Args:
            rand_generator : agent's random generator to keep consistency and reproducibility
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []
        
        for i in range(len(q_values)):
        
            if q_values[i] > top:
                top = q_values[i]
                ties = []
            
            if q_values[i] == top:
                ties.append(i)
        
        return rand_generator.choice(ties)
    
    def choose_action_egreedy(self, rand_generator, state):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
        rand_generator : agent's random generator to keep consistency and reproducibility
            state (List): coordinates of the agent (two elements)
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """
        if rand_generator.rand() < self.epsilon:
            action = rand_generator.choice(self.actions)
        else:
            values = self.q_values[state]
            action = self.argmax(values)

        return action
