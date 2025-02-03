Play the Ms.Pacman game where the objective is to eat as many red squares as possible while avoiding the ghosts, with 3 lives to spare. 

The agent plays the game autonomously using Q-Learning to describe the state action pairs throughout the game. The Q values are trained using a Reinformcement Learning Q-net neural network which takes the states as input and the Q values for each action as output. In this method the Q-net is used to link each state vector to each action.

Run the demo.ipynb file to play the game with the previously trained Q-net weights, and discover how the agent was trained. 

Q-Learning: 
The Q value is some quality score describing how good of an action is at a certain state. It is defined for each avaible action at a specific state using the Bellman equation:

Q(S_t, A_t) <--- Q(S_t, A_t) + alpha * [R_{t+1} + gamma * max_{A}(Q(S_{t+1}, A)) - Q(S_t, A_t)]

Where alpha is the learning rate, gamma is the discount factor that dictates the importance of the future Q values and potential rewards.