#Architecture

Documentation in progress. In the meantime please refer to the [README](../../README.md)

## System architecture

The system consists of 
* Command Line Interface (CLI): An interface to add/start/stop tasks.
* Scheduler: Starts and stops the AutoML tasks.
* Generate/Train/Evaluate: The subtasks that comprise the AutoML task: network generation (via an AutoML algorithm), training and evaluation.
* Data stores: SQL database and key-value store

## Execution modes
Set in the system.json config file.

System execution modes:
* Single host: 
* Multiple hosts with concurrent training
* Multiple hosts with concurrent and distributed training

Task execution modes:
* Service: Used for fast response time 
* Library: Simplicity
* RuntoCompletion: Batch jobs, avoid state maintainance across runs

## Datastores

* SQL database
* Key-value store
* Filesystem
