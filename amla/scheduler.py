#Copyright 2018 Cisco Systems All Rights Reserved
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
""" AMLA Scheduler service
"""

import os
import json
from time import sleep
from flask import Flask, request

from common.task import Task
from common.schedule import ScheduleMem
from common.schedule import ScheduleDB


def start_database_listener(args):
    scheduler.poll_tasks()

class Scheduler(Task):
    """AMLA Scheduler
    Maintains a task list (schedule of tasks to be run)
    Implements interface to add/delete/update/read tasks and return tasks results
    Scheduling is event driven:
        The FE/CLI puts tasks in the schedule via the scheduler API
        The scheduler pulls tasks from the schedule and executes them
        On compeltion, the task calls the scheduler's put_results callback API to report results
        The scheduler may add new tasks to the schedule based on the results 
        (e.g.) a generate task may trigger the execution of multiple train tasks
    """

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(self.base_dir)
        self.name = 'scheduler'
        self.data_dir = self.base_dir + "/data/"
        self.app = app
        self.stop = False
        if self.sys_config["schedule"] == "database":
            self.schedule = ScheduleDB(self.sys_config)
        else:
            self.schedule = ScheduleMem()

        self.task_config_key = None
        self.task_config = None

    def __del__(self):
        pass

    def stop_scheduler(self):
        """Stop all tasks currently being scheduled"""
        for service in self.services:
            if service == self.name:
                continue
            self.terminate_service(service)

    def main(self):
        #Start thread to poll database
        #Main thread listens for REST calls
        #Spawned thread polls schedule for new tasks to schedule
        result = self.start_thread(start_database_listener)
        self.host = self.sys_config['host'][self.name]
        self.port = self.sys_config['port'][self.name]
        self.app.run(host=self.host, port=self.port)

    def poll_tasks(self):
        while not self.stop:
            task = self.get_task();
            if task != None:
                #print("Starting task"+str(task))
                self.start_task(task)
            else:
                #print("Nothing to schedule")
                sleep(1)

    def add_task(self, task):
        """Add task to schedule"""
        task = self.schedule.add(task)
        return task

    def get_task(self):
        """Get next task to schedule"""
        task = self.schedule.get_next()
        return task

    def delete_task(self, task):
        """Delete task from schedule"""
        self.schedule.delete(task)

    def update_task(self, task):
        """Event based scheduling
        Find new tasks, based on results
        Schedule new tasks
        """

        #If the state of the task is waiting, 
        #then the task need more tasks to comple 
        if task['state'] == "waiting":
            task['waiting_for'] = []
            for t in task['new_tasks']:
              t['iteration'] = task['iteration']
              new_task  = self.schedule.add(t)
              task['waiting_for'].append(new_task['task_id'])
        if "new_tasks" in task:
            del task["new_tasks"]
        self.schedule.update(task)
        return 0

    def get_tasks(self):
        """Get a list of all tasks in schedule"""
        tasks = self.schedule.get_all()
        return json.dumps(tasks)

    def start_task(self, task):
        """Start an AMLA task
        An AMLA task consists of train, evaluate and generate subtasks
        The task is specified through the task config file.
        """

        if 'config' not in task or task['config'] == '':
            #Start task may be called with just the task_id set, or with the 
            #full task structure
            task = self.schedule.get(task)
        self.task_config_key = task['config']
        self.task_config = self.read(self.task_config_key)
        if self.task_config == None:
            print("Check config file and attempt again")
            return
        self.arch_name = self.task_config["parameters"]["arch_name"]
        mode = self.task_config["parameters"]["mode"]
        steps = self.task_config["parameters"]["steps"]
        iterations = self.task_config["parameters"]["iterations"]
        eval_interval = self.task_config["parameters"]["eval_interval"]

        task['state'] = 'running'
        self.schedule.update(task)
        if 'iteration' not in task:
            task['iteration'] = 0

        if self.task_config["parameters"]["mode"] == "construct":
                self.generate(task, self.task_config["parameters"]["controller"])
        elif self.task_config["parameters"]["mode"] == "train":
                self.train_eval(task)
        return 0

    def stop_task(self, task):
        self.stop = True
        return 0


    def generate(self, task, controller): #iteration):
        """Start the Generate subtask
        The Generate subtask generates a new network, using results from
        previous iteration of train/evaluate
        """
        iteration = task['iteration']
        if self.sys_config['exec']['generate'] == "service":
            pass
        elif self.sys_config['exec']['generate'] == "library":
            print("Error: Train and generate libraries not supported yet")
            print("Set the mode to service in the system.json file ")
            exit()
            #generate = Generate(self.base_dir, self.task_config_key)
            #generate.run()
        elif self.sys_config['exec']['generate'] == "process":
            args = [ '--base_dir='+ self.base_dir, \
                '--config='+ self.task_config_key, \
                "--task='"+ json.dumps(task)+"'"]
            print("Starting generation. Iteration: "+str(iteration))
            self.exec_process('generate/' + controller +'/generate.py', args) 
            print("Completed generation. Iteration: "+str(iteration))
        elif self.sys_config['exec']['eval'] == "deployer":
            pass
        else:
            print("Error: Invalid execution mode specified in configuration")
            print("Should be either library, process, service, or deployer")
            exit(-1)

    def train_eval(self, task):
        eval_interval = self.task_config["parameters"]["eval_interval"]
        steps = self.task_config["parameters"]["steps"]
        for step in range(int(eval_interval), int(steps)+int(eval_interval),\
                 int(eval_interval)):
            task['steps'] = step
            self.train(task)
            self.eval(task)

    def train(self, task): #steps, redirect='>>', iteration=0):
        """Start the Train subtask
        The Train subtask trains a network generated by the Generate subtask
        """
        if task['steps'] > self.task_config["parameters"]['eval_interval']:
            redirect = '>>'
        else:
            redirect = '>'

        #Use the config generated by the last iteration of generate
        mode = self.task_config["parameters"]["mode"]
        if mode == "construct":
            config_key = "results/"+self.arch_name+"/"+str(iteration)+"/config/config.json"
        else: 
            config_key = self.task_config_key
        if self.sys_config['exec']['train'] == "service":
            print("Error: Train and generate services not supported yet")
            print("Set the mode to process in the system.json file ")
            exit()
        elif self.sys_config['exec']['train'] == "library":
            print("Error: Train and generate libraries not supported yet")
            print("Set the mode to process in the system.json file ")
            exit()
            #train = Train(self.base_dir, config_key)
            #train.run()
        elif self.sys_config['exec']['train'] == "process":
            #TODO: Fix the result write method
            #Pass results file as arg to train
            #Fix tf results write
            print ("Training in progress. Iteration: "+str(task['iteration']))
            results_key = "results/"+self.arch_name+"/"+str(task['iteration'])+"/train/results.train.log"
            self.write(results_key, {})
            self.exec_process('train', ['--config='+\
                config_key, '--base_dir='+self.base_dir,\
                "--task='"+ json.dumps(task)+"'" \
                , " 2"+redirect+results_key])
        elif self.sys_config['exec']['train'] == "deployer":
            #TODO: Kubeflow
            pass
        else:
            print("Error: Invalid execution mode specified in configuration")
            print("Should be either library, process, service, or deployer")
            exit(-1)

    def eval(self, task):
        """Start the Evaluate subtask
        The Evaluate subtask evaulates a network trained by the Train subtask
        """
        if task['steps'] > self.task_config["parameters"]['eval_interval']:
            redirect = '>>'
        else:
            redirect = '>'
        mode = self.task_config["parameters"]["mode"]
        if mode == "construct":
            config_key = "results/"+self.arch_name+"/"+str(iteration)+"/config/config.json"
        else: 
            config_key = self.task_config_key
        if self.sys_config['exec']['evaluate'] == "service":
            pass
        elif self.sys_config['exec']['evaluate'] == "library":
            pass
        elif self.sys_config['exec']['evaluate'] == "process":
            print ("Evaluation in progress. Iteration: "+str(task['iteration']))
            results_key = "results/"+self.arch_name+"/"+str(task['iteration'])+"/evaluate/results.eval.log"
            self.write(results_key, {})
            self.exec_process('evaluate', ['--config='+\
                config_key, '--base_dir='+self.base_dir,\
                "--task='"+ json.dumps(task)+"'", \
                "2"+redirect+results_key])
        elif self.sys_config['exec']['evaluate'] == "deployer":
            #TODO: Kubeflow
            pass
        else:
            print("Error: Invalid execution mode specified in configuration")
            print("Should be either library, process, service or deployer")
            exit(-1)

app = Flask("scheduler")
@app.route('/api/v1.0/tasks/add', methods=['POST'])
def add_task():
    #global sched
    config = json.loads(request.data.decode())
    return json.dumps(scheduler.add_task(config))

@app.route('/api/v1.0/tasks/delete', methods=['POST'])
def delete_task():
    task = json.loads(request.data.decode())
    scheduler.delete_task(task)
    return json.dumps({"result": "OK"})

@app.route('/api/v1.0/tasks/update', methods=['POST'])
def update_task():
    task = json.loads(request.data.decode())
    result = scheduler.update_task(task)
    return json.dumps({"result": str(result)})

@app.route('/api/v1.0/tasks/get', methods=['GET'])
def get_tasks():
    return json.dumps(scheduler.get_tasks())

def start_scheduler_task(task):
    scheduler.start_task(json.loads(task))

@app.route('/api/v1.0/tasks/start', methods=['POST'])
def start_task():
    task = json.loads(request.data.decode())
    print("Scheduler task:"+str(task))
    print("Scheduler task:"+str(task['task_id']))
    result = scheduler.start_thread(start_scheduler_task, task)
    return json.dumps({"result": str(result)})

@app.route('/api/v1.0/tasks/stop', methods=['POST'])
def stop_task():
    #global sched
    task = json.loads(request.data.decode())
    result = scheduler.stop_task(task)
    return json.dumps({"result": str(result)})

@app.route('/api/v1.0/scheduler/stop', methods=['POST'])
def stop_scheduler():
    #global sched
    print("Stopping scheduler")
    result = scheduler.stop_scheduler()
    return json.dumps({"result": str(result)})


@app.route('/api/v1.0/scheduler/results', methods=['POST'])
def put_results():
    task = json.loads(request.data.decode())
    result = scheduler.start_thread(put_results, task)
    return json.dumps({"result": str(result)})

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--base_dir', help='Base directory')
    #args = parser.parse_args()
    scheduler = Scheduler()
    scheduler.run()
