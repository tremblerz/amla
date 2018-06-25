# Parallel training on AMLA

Parallel training is the training of multiple generated networks in 
parallel on multiple nodes.

## Problem
Some AutoML algorithms (e.g. Neuro Evolution) generate several candidate 
networks after one iteration. These networks need to be trained and 
evaluated and the best performing network has to be identified. Parallel 
execution of triaing of different networks reduces the tiem to find the 
best network.

##Solution

An AMLA pod will consist of multiple nodes, each running a scheduler (and possibly 
generate/train/evaluate as services or as run to completion processes).
Parallel training feature will be implemented as follows:

 * To run an AutoML algorithm, with parallel training, the CLI/FE will add 
a generate task to the database (directly or via a scheduler). The task 
information will indicate that algorithm needs parallel training.

 * One free scheduler (the first to acquire the task from the database)
 will pull the task from the database and start a generator.

 * The generator will generate multiple networks and generate one config 
 for each network. It will make a REST call to the scheduler to indicate 
 completion of the first iteration of generation
 results. It will poll the scheduler until training/evaluation of these networks is complete

 * The scheduler will generate one new training/evaluation task for each 
 network config and add to the database

 * Free schedulers will pull training/evaluation tasks from the database and 
 execute them

* The train/evaluation tasks will make a REST call to a scheduler when complete.

 * The generator continues to poll its scheduler and will continue the next iteration, once it finds 
 all training/evaluation tasks are complete.

## Alternative methods

## Limitations
* The Generate task is not parallelized

## Difference between parallel training and distributed training
* Parallel training: Training of multiple networks in parallel - one network per node
* Distributed training: Training of a single network across multiple GPUs

