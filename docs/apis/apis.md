# APIs


## Scheduler API

### CLI/FE -> Scheduler 
* GET /api/v1.0/scheduler/stop
* PUT /api/v1.0/scheduler/start
* PUT /api/v1.0/scheduler/tasks/add
* PUT /api/v1.0/scheduler/tasks/delete
* PUT /api/v1.0/scheduler/tasks/get
* PUT /api/v1.0/scheduler/tasks/start
* PUT /api/v1.0/scheduler/tasks/get

### Generate -> Scheduler
* PUT /api/v1.0/scheduler/generate/results/add

### Train -> Scheduler
* PUT /api/v1.0/scheduler/train/results/add

### Evaluate -> Scheduler
* PUT /api/v1.0/scheduler/evaluate/results/add
