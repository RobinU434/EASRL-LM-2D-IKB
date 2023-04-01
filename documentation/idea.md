This class is about sequential decision making
every agent controls a certain part of the arm
e.g.: 4 agents, every agent controls 3 joints (agent1: joint 0 to 2, ...)
input for every agent:

- target_position
- origin position from the predecessor / not really possible because we putting out a parameterized distribution -> input mu and std or calculating the 
- angels from the joints that the agent is controlling (build a second version where the input space is way bigger)
output from every agent:
- angels where to set joints

Compare performance with agent where the input contains all angles from the arm


Compare performance with agent where each agent controls a different amount of joints... 12 or 24 could be  a good number
e.g for 12 joints

- 2 agents with 6 joints each
- 3 agents with 4 joints each
- 4 agents with 3 joints each
- 6 agents with 2 joints each


Compare Performance with LSTM Actor with same inputs but each LSTM step controls ONE joint
