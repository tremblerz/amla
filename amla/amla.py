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
""" AMLA CLI implementation
"""

import cmd
import os

from common.task import Task
from scheduler import Scheduler

class AmlaCLI(cmd.Cmd):
    """ AMLA CLI interface functions
    - Start/Stop of scheduler
    - CRUD operations on tasks
    - Start/Stop of tasks
    """

    intro = 'AMLA: Auto ML frAmework for Neural Nets \n \
        Command Line Interface. Type help or ? to list commands.'
    prompt = 'amla#'

    def __init__(self):
        super().__init__()
        base_dir = os.path.dirname(os.path.realpath(__file__))
        self.amla = Amla(base_dir)
        self.amla.start_scheduler(None)

    #AMLA cmds
    def emptyline(self):
        pass
    def do_start_scheduler(self, arg):
        'Start scheduler'
        self.amla.start_scheduler(self.parse(arg))
    def do_stop_scheduler(self, arg):
        'Stop scheduler'
        self.amla.stop_scheduler(self.parse(arg))
    def do_add_task(self, arg):
        'Add a task to the scheduler task list'
        self.amla.add_task(self.parse(arg)[0])
    def do_delete_task(self, arg):
        'Set configuration'
        self.amla.delete_task(self.parse(arg)[0])
    def do_get_tasks(self, arg):
        'Get a list of the tasks in the scheduler task list'
        print(self.amla.get_tasks(arg))
    def do_start_task(self, arg):
        'Start a task'
        self.amla.start_task(self.parse(arg)[0])
    def do_stop_task(self, arg):
        'Stop a task'
        self.amla.stop_task(self.parse(arg))
    def do_exit(self, arg):
        'Exit'
        self.amla.stop_scheduler(arg)
        exit(0)
    def do_quit(self, arg):
        'Exit'
        self.amla.stop_scheduler(arg)
        exit(0)

    def parse(self, arg):
        return arg.split()

class Amla(Task):
    """ AMLA CLI backend functions
    - Start/Stop of scheduler
    - CRUD operations on tasks
    - Start/Stop of tasks
    """
    def __init__(self, base_dir):
        self.name = 'amla'
        super().__init__(base_dir)
        self.base_dir = base_dir
        self.scheduler = None

    def start_scheduler(self, args):
        self.exec_process_async('scheduler', None)

    def stop_scheduler(self, args):
        parameters = {"op": "POST"}
        self.send_request("scheduler", "scheduler/stop", parameters)
        self.terminate_service('scheduler')

    def add_task(self, config):
        parameters = {"config": config, "op": "POST"}
        task = self.send_request("scheduler", "tasks/add", parameters)
        print("Added task: " + str(task) + " to schedule.")

    def delete_task(self, task_id):
        parameters = {"task_id": int(task_id), "op": "POST"}
        self.send_request("scheduler", "tasks/delete", parameters)

    def get_tasks(self, arg):
        parameters = {'op': 'GET'}
        tasks = self.send_request("scheduler", "tasks/get", parameters)
        return tasks

    def start_task(self, task_id):
        parameters = {"task_id": int(task_id), "op": "POST"}
        self.send_request("scheduler", "tasks/start", parameters)

    def stop_task(self, task_id):
        parameters = {"task_id": task_id, "op": "POST"}
        self.send_request("scheduler", "tasks/stop", parameters)

if __name__ == "__main__":
    amla = AmlaCLI()
    amla.cmdloop()
