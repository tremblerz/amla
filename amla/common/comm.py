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
"""Base communication class
"""

import json
import requests

class Comm():
    """Base communication class
    Abstracts message sending/receive methods (REST client/?)
    """
    def __init__(self, sys_config):
        self.sys_config = sys_config

    def __del__(self):
        pass

    def send_request(self, service, operation, parameters):
        """Makes a REST call (GET/POST)
        """
        host = self.sys_config['host'][service]
        port = self.sys_config['port'][service]
        url = "http://" + host + ":" + str(port) + "/api/v1.0/"+operation
        if parameters['op'] == 'GET':
            del parameters['op']
            response = requests.get(url, json=parameters)
        elif parameters['op'] == 'POST':
            del parameters['op']
            response = requests.post(url, json=parameters)
        if not response.ok:
            print("Comm: Error sending message")
        return json.loads(response.text)
