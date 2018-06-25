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
"""Base task class
"""
import os
import signal
import json
import threading
from subprocess import Popen
from abc import ABCMeta
from abc import abstractmethod

from common.store import Store
from common.comm import Comm

class Task:
    """Base task class. All Tasks derive from this class
    Abstracts
    - deployment methods (service/thread/process/library/deployer)
    - request processing method (threading/?)
    - microservice library (Flask/?),
    - persistent key-value store (file system/?)
    - message sending/receive methods (REST client/?)
    """
    __metaclass__ = ABCMeta
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.sys_config_key = "configs/system.json"
        self.get_sys_config()
        self.store = Store(base_dir, self.sys_config)
        self.comm = Comm(self.sys_config)
        self.children = {}
        self.name = ""
        self.app = None
        self.threads = []
        self.host = ""
        self.port = 0
    def __del__(self):
        pass

    def get_sys_config(self):
        key = self.base_dir + "/" + self.sys_config_key
        with open(key, 'r') as fread:
            self.sys_config = json.load(fread)
        self.services = self.sys_config['services']
        self.exes = self.sys_config['exes']

    @abstractmethod
    def main(self):
        """Main function
        Defined in the child class
        """
        pass

    def run(self):
        """Run this task
        The task can be run as a service or as a library
        In either case the class' main function gets called
        """
        if self.sys_config['exec'][self.name] == "service":
            self.host = self.sys_config['host'][self.name]
            self.port = self.sys_config['port'][self.name]
            self.app.run(host=self.host, port=self.port)
        elif self.sys_config['exec'][self.name] == "library":
            self.main()
        elif self.sys_config['exec'][self.name] == "process":
            self.main()
        else:
            print("Error: Invalid execution mode specified in configuration")
            print("Should be either module, runtocompletion or service")
            exit(-1)

    def start_thread(self, func, data):
        """Start a thread within this task
        """
        sdata = json.dumps(data)
        thread = threading.Thread(target=func, args=[sdata])
        self.threads.append(thread)
        thread.start()

    def stop_thread(self):
        """Stop a running thread within this task
        """
        #TODO
        return True

    def exec_process(self, process, args=None):
        """Fork a new process to run a new task, blocking
        """
        if process in self.sys_config['exes']:
            name = self.sys_config['exes'][process]
        else:
            name = process
        sargs = " ".join(args) if args is not None else ""
        cmd = "python " + self.base_dir + "/" + name + " " + sargs
        os.system(cmd)

    def exec_process_async(self, process, args):
        """Fork a new process to run a new task, non blocking
        """
        if process in self.sys_config['exes']:
            name = self.sys_config['exes'][process]
        else:
            name = process

        sargs = [] if args is None else args
        cmd = ["python", self.base_dir + "/"+name] +sargs
        self.children[process] = Popen(cmd)

    def terminate_process(self, process):
        """Terminate a running task
        """
        #Todo: Send SIGTERM and wait. Send SIGKILL if child does not terminate
        if process in self.children:
            self.children[process].kill()

        #TODO: Poll
        #Will kill all running process on server
        #TODO: get PIDs from persistent store
        name = self.sys_config['exes'][process]
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            os.kill(int(pid), signal.SIGKILL)

    def send_request(self, service, operation, parameters):
        """Send a request to a task running as a service
        """
        return self.comm.send_request(service, operation, parameters)

    def read(self, key):
        """Read a file from the persistent store
        """
        return self.store.read(key)

    def write(self, key, data):
        """Write a file to the persistent store
        """
        return self.store.write(key, data)
