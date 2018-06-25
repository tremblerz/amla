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
class Schedule():
    """
    Implements a schedule of tasks.
    Currently a FIFO queue, stored in memory implemented using python's deque
    Note: Not safe for concurrent access
    A task is in one of 4 states: init, running, paused, complete
    TODO:
        Abstract over a persistent store that supports concurrency
    """
    def __init__(self):
        self.tasks = collections.deque()
        self.nexttaskid = 0
        return

    def add(self, config):
        """ Add a task to the schedule
        """
        taskid = self.nexttaskid
        task = {'taskid': taskid, 'config': config['config'], 'state': 'init'}
        self.tasks.append(task)
        self.nexttaskid += 1
        return task

    def update(self, task):
        for elem in self.tasks:
            if elem['taskid'] == task['taskid']:
                for key in task:
                    elem[key] = task[key]
        return

    def delete(self, task):
        if  not task:
            return -1
        for elem in self.tasks:
            if elem['taskid'] == task['taskid']:
                break
        return task['taskid']

    def get(self, task):
        elem = None
        for elem in self.tasks:
            if elem['taskid'] == task['taskid']:
                break
        return elem

    def get_next(self):
        """Get the next task to be scheduled
         Currently uses a FIFO queue
        """
        return self.tasks.popleft()

    def get_all(self):
        return list(self.tasks)
