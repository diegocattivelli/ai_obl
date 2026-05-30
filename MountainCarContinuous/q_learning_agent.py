import random
import numpy as np
import matplotlib.pyplot as plt


class QLearningAgent:

    # el agente neecsita conocer la tabla de Q
    # tenemos que tener posicion, velocidad y acción
    def __init__(self, states, actions):
        self.q= np.zeros((states, actions))

    # nos llega observacion que es el estado donde estamoms, me quedo con la acción de mejor utilidad, que me maximiza
    def next_action(self, obs):
        state= obs
        action = np.argmax(self.q[state])
        return action
    

    def _epsilon_greedy_policy(state, Q, epsilon):
        explore = np.random.binomial(1, epsilon)
        if explore:
            action = env.action_space.sample()
            print('explore')
        # exploit
        else:
            action = np.argmax(Q[state])
            print('exploit')
        
        return action

    #entrenar el agente, mapear lo de la diapostiva de como cargar la tabla de Q
    # acá hacemos muchos episodios, ejecutar muchas veces nuestro agente 
    #devuelvo la tabla Q y las recompensas.
    def train_agent(self, env, episodes=1000, epsilon=.9, gamma=.9, alpha=.99):
        current_state, _= env.reset()
        rewards = []
        for episode in range (episodes):
            done = False
            rewards_episode=0
            while not done:
                action = self._epsilon_greedy_policy(current_state, self.q, env, epsilon)
                next_state,reward, done, _, _= env.step(action)
                self.q[current_state][action] = self.q[current_state][action]+ alpha*(reward + gamma*np.max(self.q[next_state])-self.q[current_state][action])
                current_state=next_state
                rewards_episode+=reward
            rewards.append(rewards_episode)
            current_state,_= env.reset()
        return self.q
    
    #toma la accion máxima, a partir de la tabla de q va a pbtener la accion que maximiza la utilidad
    #depsues de entrenar al agente realizar un test en el cual el agente elije la accion optima basada en la tabla q entrenada
    def test_agent(self, env, episodes=10):
        #acá solo tomamos la mejor acción, la que maximiza la utilidad
        for episode in range(episodes):
            current_state, _= env.reset()
            done=False
            while not done:
                action= self.next_action(current_state)
                next_state,reward, done,_,_=env.step(action)
                current_state = next_state

