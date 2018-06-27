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
"""
Base store class
"""

import os
import json

class Store:
    """
    Base store class
    Abstracts persistent key-value store (file system/?)
    """
    def __init__(self, base_dir, sys_config):
        self.sys_config = sys_config
        self.base_dir = base_dir

    def __del__(self):
        pass

    def read(self, key):
        if self.sys_config['persistent'] == 'filesystem':
            key = self.base_dir + "/" + key
            try:
                with open(key, 'r') as fread:
                    raw = fread.read();
                    try: 
                        value = json.loads(raw)
                    except:
                        value = raw
                    fread.close()
                return value
            except IOError:
                print("Error: Could not access key: " + key)
                return None
        else:
            print("Error: Unsupported persistent store")
            exit(-1)

    def write(self, key, data):
        if self.sys_config['persistent'] == 'filesystem':
            key = self.base_dir + "/" + key
            if not os.path.exists(os.path.dirname(key)):
                try:
                    os.makedirs(os.path.dirname(key))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                       raise
            try:
                with open(key, 'w') as fwrite:
                    #TODO: Cleanup
                    if type(data) is dict or type(data) is list:
                        fwrite.write(json.dumps(data, indent=4, sort_keys=True))
                    else:
                        fwrite.write(data)
                    fwrite.close()
                return True
            except IOError:
                print("Error: Could not access key: " + key)
                return None
        else:
            print("Error: Unsupported persistent store")
            exit(-1)
