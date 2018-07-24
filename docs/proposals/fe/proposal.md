# Proposal for AMLA front end


## Organization:
The front end will be designed as widgets for
* Task management
* Results visualization
* ?
The widgets will be written in vue.js using the CoreUI template
for UI elements.
The widgets wll use the scheduler API to get/put data/post requests.

## Task management:

### Add/delete/list/edit tasks
* Allow a user to add, delete and get information about a task
* Some of the APIs needed are currently supported by the scheduler, but may need fixes

### Show config
* Allow a user to view the config file for a task
* Needs scheduler API support to get config file for a task

## Results visualization:

### Accuracy:
* Show accuracy as task progresses
* Needs API support to retrieve results files

### Resource statistics:
* Show statistics on resource usage for task (wall clock time, cpu, cost)
* Needs API support to retrieve multiple config files

### Network generation:
* Show network generation progress over time
* Needs API support to retrieve multiple config files
* Needs js/d3 support to visualize networks
