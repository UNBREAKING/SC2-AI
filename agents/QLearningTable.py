import numpy as np
import pandas as pd
import math

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list?
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose the best action from q-table
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    # Q-learning implementation

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # Make q-table and select max value

        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # update

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
