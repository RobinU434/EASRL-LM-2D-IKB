each folder is named after the number of joints 
each folder contains 3 sub folders

- train with actions and targets
- val with actions and targets
- test with actions and targets

there are different kinds of actions:

- random: random start before ccd
- const: constant start at 0s before ccd
- relative_uniform: relative / strategic actions sampled from [-1, 1]
- relative_tanh: relative/ strategic actions sampled from normal and than put into an tanh function

Note:
The actions and targets where stored with an index.
The target filed contains a third dimension. This is a result from the CCD implementation which only accepts a 3 dimensional vector.
