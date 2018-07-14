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
"""Task Schedule
"""

import collections
import operator
import json

class Schedule():
    """
    Base class for schedule
    TODO: Make ABC
    A task is in one of 3 states: init, running, waiting, complete
    """
    def __init__(self):
        return
    
    def get_next_task(self, tasks):
          """ 
          Scheduling algorthm
          Find oldest task where state is "init"
          or "waiting" and "waitingfor" are completed
          TODO: Topological sort of task graph
          """
          if len(tasks) == 0:
              return None
          else:
              schedulable =[]
              for task in tasks:
                  if task["state"] == "init":
                     schedulable.append(task)
                  elif task["state"] == "waiting":
                     schedule = True
                     for task_id in task["waiting_for"]:
                         #Search for task in schedule
                         #TODO: Create an index
                         for t in tasks:
                             if task_id == t["task_id"]:
                                 if t["state"] != "complete":
                                     schedule = False
                                     break
                     if schedule == True:
                         schedulable.append(task)
                         #After the tasks complete, then the iteration needs to be 
                         #increased before this task is resumed 
                         #TODO: Specific to generate tasks
                         #Move elsewhere?
                         task['iteration']+=1
                         task['state'] = 'running'
                         task['waiting_for'] = [];
              task = None
              #TODO: Sort schedulable based on task id
              if len(schedulable) > 0:
                  schedulable.sort(key=operator.itemgetter('task_id'))
                  task = schedulable[0]
                  task['state'] = 'running'
                  task['waiting_for'] = [];
              return task

class ScheduleMem(Schedule):
    """
    Implements a schedule of tasks stored in memory
    Can be used only when AMLA is used in single host mode
    Currently a FIFO queue, stored in memory implemented using python's deque
    Note: Not safe for concurrent access
    TODO:
    """
    def __init__(self):
        #TODO: Assert in single host mode
        self.tasks = collections.deque()
        self.nexttask_id = 0
        return

    def add(self, t):
        """ Add a task to the schedule
        """
        task_id = self.nexttask_id
        task = {'task_id': task_id, 'config': t['config'], 'state': 'init'}
        if 'iteration' in t:
           task['iteration'] = t['iteration']
        else:
           task['iteration'] = 0
        self.tasks.append(task)
        self.nexttask_id += 1
        return task

    def update(self, task):
        for elem in self.tasks:
            if elem['task_id'] == task['task_id']:
                for key in task:
                    elem[key] = task[key]
        return

    def delete(self, task):
        if  not task:
            return -1
        for elem in self.tasks:
            if elem['task_id'] == task['task_id']:
                break
        return task['task_id']

    def get(self, task):
        elem = None
        for elem in self.tasks:
            if elem['task_id'] == task['task_id']:
                break
        return elem

    def get_next(self):
        """Get the next task to be scheduled
         Currently uses a FIFO queue
        """
        if len(self.tasks) == 0:
            return None
        else:
            task = self.get_next_task(self.tasks)
            return task

    def get_all(self):
        return list(self.tasks)

class ScheduleDB(Schedule):
    """
    Implements a schedule of tasks stored in a DB
    Currently uses mysql, with transactions to support 
    concurrent schedulers
    """
    def __init__(self, sys_config):
        import MySQLdb
        host = sys_config["database"]["host"]
        user = sys_config["database"]["user"]
        passwd = sys_config["database"]["password"]
        db = sys_config["database"]["db"]
        self.db = MySQLdb.connect(host=host,
                     user=user,
                     passwd=passwd,
                     db=db)

        self.cur = self.db.cursor()
        query = "CREATE TABLE IF NOT EXISTS schedule ( \
            task_id INT(11) NOT NULL AUTO_INCREMENT, \
            config VARCHAR(1024) DEFAULT NULL, \
            state VARCHAR(32) DEFAULT 'init', \
            steps INT(11) DEFAULT 0, \
            iteration INT(11) DEFAULT 0, \
            waiting_for VARCHAR(1024) DEFAULT NULL, \
            PRIMARY KEY(task_id)) ENGINE=InnoDB;"
        self.cur.execute(query)
        self.db.commit()
        return

    def __del__(self):
        self.db.close()

    def add(self, task):
        """ Add a task to the schedule
        """
        #Task_id is Find task_id, increment and add new task
        #Must be atomic
        iteration = 0
        if 'iteration' in task:
            iteration = task['iteration']
        query = "INSERT INTO schedule (config, iteration, state, waiting_for) VALUES \
             ('"+task['config']+"', "+str(iteration)+", 'init', '[]');"
        self.cur.execute(query)
        self.db.commit()

        task_id = self.cur.lastrowid
        task['task_id'] = task_id
        return task

    def update(self, task):
        #TODO
        if 'waiting_for' not in task:
            task['waiting_for'] = []
        query = "UPDATE schedule set state= '"+task['state']+"', waiting_for='"\
            +json.dumps(task['waiting_for'])+"' WHERE task_id = "+str(task['task_id'])+";"
        self.cur.execute(query)
        self.db.commit()
        return

    def delete(self, task):
        query = "DELETE FROM schedule WHERE task_id = "+str(task['task_id'])+";"
        self.cur.execute(query)
        self.db.commit()
        return task['task_id']

    def get(self, task):
        query = "SELECT task_id, config, state FROM schedule WHERE task_id = "+str(task['task_id'])+";"
        self.cur.execute(query)
        row = self.cur.fetchone()
        task = {"task_id": row[0], "config": row[1], "state": row[2]}
        return task

    def get_next(self):
        """Get the next task to be scheduled
        Gets the task with the least task_id (oldest task) whose state is 'init'
        """
        self.db.autocommit(False)
        self.cur.execute("START TRANSACTION;")
        task = None
        try:
            query = "SELECT task_id, config, state, iteration,  waiting_for FROM schedule \
                WHERE state='init' OR state='waiting';"
            self.cur.execute(query)
            rows = self.cur.fetchall()
            if len(rows) == 0:
                #No tasks to schedule
                return None
            tasks = []
            for row in rows:
                task = {"task_id": row[0], "config": row[1], "state": row[2],\
                    "iteration": int(row[3]), "waiting_for": json.loads(row[4])}
                tasks.append(task)
            task = self.get_next_task(tasks)
            if task == None:
                self.db.rollback()
                return None
                
            query = "UPDATE schedule set state = 'running', waiting_for='[]', \
                  iteration='"+str(task['iteration'])+"'  WHERE task_id = "+str(task['task_id'])+";"
            
            self.cur.execute(query)
            self.db.commit()
        except:
            print("Error: Could not commit transaction. Rolling back")
            self.db.rollback()
        return task


    def get_all(self):
        query = "SELECT task_id, config, state FROM schedule;"
        self.cur.execute(query)
        rows = self.cur.fetchall()
        tasks = []
        for row in rows:
            task = {"task_id": row[0], "config": row[1], "state": row[2]}
            tasks.append(task)
        return tasks
