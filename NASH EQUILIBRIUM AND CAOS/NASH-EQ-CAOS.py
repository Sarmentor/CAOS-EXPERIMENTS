#please relate nash equilibrium with caos theory. Present pyhton code that could be used to mix both theories in an experiment
#Nash equilibrium and chaos theory are two distinct concepts in game theory and mathematics, with different underlying principles and applications. However, it is possible to explore their relationship by considering the dynamics of strategic interactions and the emergence of complex behaviors.

#Nash equilibrium, named after mathematician John Nash, refers to a concept in game theory where each player's strategy is the best response to the strategies chosen by the other players, resulting in a stable outcome. Chaos theory, on the other hand, studies complex and unpredictable behavior in deterministic systems, often characterized by extreme sensitivity to initial conditions.

#To incorporate elements of both theories in an experiment, we can consider a scenario with multiple players interacting in a dynamic environment. Here's an example of a simple Python code that combines both concepts:

#```python
import numpy as np
import matplotlib.pyplot as plt

# Define the game
payoff_matrix = np.array([[1, 2],
                          [1, 3]])#,
#                          [2, 0]])

# Initialize player strategies
num_players = 2
strategies = np.random.uniform(0, 1, size=(num_players,))
print(strategies)

# Set chaos parameters
chaos_param = 2.48
#chaos_param = 4.8

num_iterations = 100

# Iterate the game
history = []
for t in range(num_iterations):
    # Calculate payoff for each player
    payoffs = np.dot(payoff_matrix.transpose(), strategies)
    #print(payoffs)
    
    # Update strategies based on Nash equilibrium
    best_responses = np.argmax(payoffs, axis=0)
    #print(best_responses)
    
    strategies = np.zeros_like(strategies)
    
    for i in range(num_players):
        strategies[i] = payoffs[best_responses]

        # Introduce chaos by applying a logistic map
        strategies[i] = chaos_param * strategies[i] * (1 - strategies[i])
        
        #NO CAOS
        #strategies[i] = strategies[i]
        
        # Store strategies for visualization
    history.append(strategies.copy())

# Plot the evolution of strategies
history = np.array(history)
plt.plot(history[:, 0], label='Player 1')
plt.plot(history[:, 1], label='Player 2')
#plt.plot(history[:, 2], label='Player 3')
print(history)
plt.xlabel('Iterations')
plt.ylabel('Strategy')
plt.legend()
plt.show()
#```

#In this code, we consider a two-player game with a payoff matrix. Each player's strategy is updated based on the Nash equilibrium concept, where they choose the best response strategy to their opponent's strategy. Additionally, chaos theory is introduced by applying the logistic map to the strategies, adding a chaotic element to the system.

#The code iterates the game for a specified number of iterations and stores the strategies at each step. Finally, it plots the evolution of the strategies over time.

#Please note that this is a simplified example meant to demonstrate the combination of Nash equilibrium and chaos theory. The specific dynamics and outcomes will depend on the specific game and chaos parameters chosen, and further customization and analysis may be required for a comprehensive experiment.