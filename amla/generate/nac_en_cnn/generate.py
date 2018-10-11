# Copyright 2018 Cisco Systems All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Generate networks using the algorithms discussed in 
https://arxiv.org/abs/1803.06744
"""

import sys
import json
import copy
import re
import random
import math
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', help='Base directory')
parser.add_argument('--config', help='Configuration file key')
parser.add_argument('--task', help='Task information')
#parser.add_argument(
#    '--lastconfig',
#    help='Configuration file key from previous iteration')
#parser.add_argument(
#    '--lastresults',
#    help='Results file key from previous iteration')
args = parser.parse_args()
sys.path.insert(0, args.base_dir)
from common.task import Task


class Generate(Task):
    """Generate class
    - Generation using EnvelopNets
    - Random network generation
    """

    def __init__(self, base_dir, config, task):
        super().__init__(base_dir)
        self.name = 'generate'
        self.arch = []
        self.task_config_key = config
        self.task_config = self.read(config)
        self.task = json.loads(task)
        self.iteration = self.task['iteration']
        self.base_dir = base_dir
        self.get_task_params()

    def get_task_params(self):
        self.algorithm = self.task_config["parameters"]["algorithm"]
        self.mode = self.task_config["parameters"]["mode"]
        self.arch_name = self.task_config["parameters"]["arch_name"]
        self.init_cell = self.task_config["init_cell"]
        self.classification_cell = self.task_config["classification_cell"]
        if self.algorithm == "envelopenet":
            self.max_filter_prune = int(
                self.task_config["envelopenet"]["max_filter_prune"])
            if 'worst_case' in self.task_config["envelopenet"]:
                self.worst_case = self.task_config["envelopenet"]["worst_case"]
            else:
                self.worst_case = False
            self.envelopecell = self.task_config["envelopenet"]["envelopecell"]
            self.layers_per_stage = self.task_config["envelopenet"]["layers_per_stage"]
            self.max_layers_per_stage = self.task_config["envelopenet"]["max_layers_per_stage"]
            self.stages = int(self.task_config["envelopenet"]["stages"])
            self.parameter_limits = self.task_config["envelopenet"]["parameter_limits"]
            self.construction = self.task_config["envelopenet"]["construction"]
        elif self.algorithm == "random":
            self.parameter_limits = self.task_config["random"]["parameter_limits"]
            self.layers_per_stage = self.task_config["random"]["layers_per_stage"]
            self.stages = int(self.task_config["random"]["stages"])
            self.num_blocks = self.task_config["random"]["num_blocks_per_stage"]
            self.blocks = (self.task_config["random"]["blocks"])
        elif self.algorithm == "deterministic":
            self.arch = self.task_config["arch"]
        else:
            print("Invalid algorithm")
            exit(-1)

    def __del__(self):
        pass

    def save_config(self, arch):
        #Generate the config file with architecture for this iteration
        config = copy.deepcopy(self.task_config)
        config["arch"] = arch
        config["parameters"]["algorithm"] = "deterministic"
        config["parameters"]["mode"] = "train"
        del config["envelopenet"]
        key = "results/" + self.arch_name + "/" + str(self.iteration) \
            + "/train/config/config.json"
        self.write(key, config)
        config["parameters"]["mode"] = "evaluate"
        key = "results/" + self.arch_name + "/" + str(self.iteration) \
            + "/evaluate/config/config.json"
        self.write(key, config)

    def main(self):
        if self.mode == "oneshot" or self.iteration == 0:
            arch = self.generate()
            self.save_config(arch)
            # TODO: Generate a oneshot config file
        else:
            prev_iteration = self.iteration - 1
            prev_arch_key = "results/" + self.arch_name + "/" + \
                str(prev_iteration) + "/train/config/config.json"
            prev_results_key = "results/" + self.arch_name + \
                "/" + str(prev_iteration) + "/train/results.train.log"
            prev_arch = self.read(prev_arch_key)["arch"]
            arch = self.construct(prev_arch, prev_results_key)
            self.save_config(arch)
        if self.sys_config['exec']['scheduler'] == "service":
             self.put_results()

    def put_results(self):
        task = self.task
        task["op"] =  "POST"
        if self.iteration == self.task_config["parameters"]["iterations"]:
            task['state'] = "complete"
        else:
            new_tasks = []
            newtask = {"task_id": "", "config":  "results/" + \
                self.arch_name + "/" + str(self.iteration) + \
                "/train/config/config.json"}
            new_tasks.append(newtask)

            #Two tasks for testing
            #Remove
            newtask = {"task_id": "", "config":  "results/" + \
                self.arch_name + "/" + str(self.iteration) + \
                "/evaluate/config/config.json"}
            #new_tasks.append(newtask)

            task['state'] = "waiting"
            task['new_tasks'] = new_tasks
        self.send_request("scheduler", "tasks/update", task)

    def generate(self):
        """ Generate a network arch based on network config params: Used for
        "oneshot" mode or the initial network when run in "construct" mode
        """
        if self.algorithm == "deterministic":
            return self.task_config["arch"]
        elif self.algorithm == "envelopenet":
            return self.gen_envelopenet_bystages()
        elif self.algorithm == "random":
            return self.gen_randomnet()
        else:
            print("Invalid algorithm")
            exit(-1)

    def construct(self, arch, samples):
        """ Construct a new network based on current arch, metrics from last 
        run and construction config params
        """
        self.arch = arch
        if self.algorithm == "envelopenet":
            return self.construct_envelopenet_bystages(samples)
        elif self.algorithm == "random":
            return self.construct_random()
        else:
            print("Invalid algorithm")
            exit(-1)

    def construct_random(self):
        # Return a random network, equivalent to a network create
        # by an envelope net generation algorithm, but with random pruning
        narch = self.gen_randomnet()
        narch = self.insert_skip(narch)
        return narch

    def construct_envelopenet_bystages(self, samples):
        arch = {"type":"macro","network":[]}
        lsamples = self.read(samples)
        nsamples = lsamples.split('\n')
        # offset_count takes care of how much a present layer gets shifted after new construction
        offset_count = [0]
        worst_case = self.worst_case
        stages = []
        stage = []
        stagecellnames = {}
        ssidx = {}
        # Cell0 is the init block
        lidx = 1
        ssidx[0] = 1
        stagenum = 0
        for layer in self.arch["network"]:
            if 'widener' in layer:
                lidx += 1
                stages.append(stage)
                stage = []
                stagenum += 1
                ssidx[stagenum] = lidx
            else:
                for branch in layer["filters"]:
                    # TODO: Add the cellname to the config file the same as
                    # cell names in the logfiles
                    cellname = 'Cell' + str(lidx) + "/" + branch
                    if stagenum not in stagecellnames:
                        stagecellnames[stagenum] = []
                    stagecellnames[stagenum].append(cellname)
                stage.append(layer)
                lidx += 1
        stages.append(stage)
        stagenum = 0
        narch = []
        #print("Stage cellnames: " + str(stagecellnames))
        #print("Stage ssidx: " + str(ssidx))
        #print("Stages: " + str(stages))
        for stage in stages:
            if self.construction[stagenum] and len(
                    stage) <= self.max_layers_per_stage[stagenum]:
                prune = self.select_prunable(
                    stagecellnames[stagenum], nsamples, worst_case=worst_case)
                #print("Stage: " + str(stage))
                #print("Pruning " + str(prune))
                nstage = self.prune_filters(ssidx[stagenum], stage, prune)
                nstage = self.add_cell(nstage)
                offset_count += [offset_count[-1]] * len(nstage)
                offset_count[-1] += 1
            else:
                nstage = copy.deepcopy(stage)
                offset_count += [offset_count[-1]] * len(nstage)
            # Do not add widener for the last stage
            self.set_outputs(nstage, stagenum)
            if stagenum != len(stages) - 1:
                nstage = self.add_widener(nstage)
                offset_count.append(offset_count[-1])
            #print("New stage :" + str(nstage))
            narch += (nstage)
            stagenum += 1

        offset_count = offset_count[1:]
        arch["network"] = narch
        narch = self.insert_skip(arch, nsamples, offset_count, dense_connect=False)
        #print(narch)
        #print("Old arch :" + str(self.arch))
        #print("New arch :" + str(narch))
        return narch

    def remove_logging(self, line):
        line = re.sub(r"\d\d\d\d.*ops.cc:79\] ", "", line)
        return line

    def filter_samples(self, samples, filter_string='MeanSSS'):
        #filter_string = 'Variance'
        filtered_log = [line for line in samples if filter_string in line]
        return filtered_log

    def get_samples(self, samples, filter_string='MeanSSS'):
        #filter_string = 'Variance'
        filtered_log = [line for line in samples if line.startswith(filter_string)]
        return filtered_log

    def get_filter_sample(self, sample):
        fields = sample.split(":")
        filt = fields[1]
        value = float(fields[2].split(']')[0].lstrip('['))
        return filt, value

    def set_outputs(self, stage, stagenum):
        init = self.init_cell
        sinit = sorted(init.keys())  # , key=init.get); #, reverse = True)
        # Input channels = output of last layer (conv) in the init
        for layer in sinit:
            #print(sinit)
            for branch in init[layer]:
                if "outputs" in init[layer][branch]:
                    inputchannels = init[layer][branch]["outputs"]
        #print("Input channels: " + str(inputchannels))
        width = math.pow(2, stagenum) * inputchannels
        #print("W : " + str(width))
        if self.parameter_limits[stagenum]:
            """ Parameter limiting: Calculate output of the internal filters such that
            overall  params is maintained constant
            """
            layers = float(len(stage))
            outputs = int((width / (layers - 2.0)) *
                          (math.pow(layers - 1.0, 0.5) - 1))
            #print("outputs : " + str(outputs))
        #print(stage)
        lidx = 0
        for layer in stage:
            #print(lidx)
            #print(len(stage) - 1)
            if "widener" in layer:
                #print("Widener")
                lidx += 1
                continue
            if lidx == len(stage) - \
                    1 or self.parameter_limits[stagenum] is False:
                #print("Setting outputs to W")
                layer["outputs"] = int(width)
            elif "filters" in layer:
                #print("Limiting outputs")
                layer["outputs"] = outputs
            lidx += 1

    def select_prunable(self, stagecellnames, samples, worst_case=False):
        samples = self.filter_samples(samples)
        measurements = {}
        for sample in samples:
            if sample == '':
                continue
            sample = self.remove_logging(sample)
            filt, value = self.get_filter_sample(sample)

            # Prune only filters in this stage
            if filt not in stagecellnames:
                continue

            if filt not in measurements:
                measurements[filt] = []
            measurements[filt].append(value)

        #print("Stage cell names" + str(stagecellnames))
        #print("Filter in samples " + str(list(measurements.keys())))
        # Rank variances, select filters to prune
        # Use last variance reading
        variances = {}
        for filt in measurements:
            variances[filt] = measurements[filt][-1]

        if worst_case:
            print("WARNING: -------GENERATING WORST CASE NET---------")
            reverse = True
        else:
            reverse = False
        svariances = sorted(variances, key=variances.get, reverse=reverse)
        # Count number of cells in each layer
        #print("All variances: " + str(variances))
        #print("Sorted variances: " + str(svariances))
        cellcount = {}
        for cellbr in variances:
            cellidx = cellbr.split("/")[0].lstrip("Cell")
            if cellidx not in cellcount:
                cellcount[cellidx] = 0
            cellcount[cellidx] += 1

        #print(cellcount)
        # Make sure we do not prune all cells in one layer
        prunedcount = {}
        prune = []
        for svariance in svariances:
            prunecellidx = svariance.split("/")[0].lstrip("Cell")
            if prunecellidx not in prunedcount:
                prunedcount[prunecellidx] = 0
            if prunedcount[prunecellidx] + 1 < cellcount[prunecellidx]:
                #print("Pruning " + svariance)
                prune.append(svariance)
                prunedcount[prunecellidx] += 1
                # Limit number of pruned cells to min of threshold * number of 
                #filters in stage and maxfilter prune
                # TODO: Move thresold to config, make configurable per stage
                # If the threshold is high enough and there are few filters in
                # stage, only one will be pruned
                threshold = (1.0 / 3.0)
                prunecount = min(self.max_filter_prune, int(
                    threshold * float(len(stagecellnames))))
                if len(prune) >= prunecount:
                    break
        if not prune:
            print(svariances)
            print("Error: No cells to prune")
            exit(-1)
        return prune

    def prune_filters(self, ssidx, stage, prune):
        #print(("Pruning " + str(prune)))
        # Generate a  pruned network without the wideners
        narch = []
        # = copy.deepcopy(self.arch);
        lidx = 0
        nfilterlayers = 0
        # for layer in self.arch:
        for layer in stage:
            if 'widener' in layer:
                lidx += 1
                continue
            #print("Layer " + str(lidx))
            #print("Arch " + str(layer))
            # narch.append(copy.deepcopy(self.arch[lidx]));
            narch.append(copy.deepcopy(stage[lidx]))
            # for filt in self.arch[lidx]["filters"]:
            for filt in stage[lidx]["filters"]:
                fidx = int(filt.lstrip("Branch"))
                for prn in prune:
                    #print("Checking " + str(prn) + " with :" +
                          #str(ssidx + lidx) + ":" + str(filt))
                    prunecidx = prn.split("/")[0].lstrip("Cell")
                    prunefidx = prn.split("/")[1].lstrip("Branch")
                    if ssidx + lidx == (int(prunecidx)) and \
                        fidx == int(prunefidx):
                        #print("Match")
                        del narch[-1]["filters"]["Branch" + str(prunefidx)]
            #print("Narc: " + str(narch[-1]))
            nfilterlayers += 1
            lidx += 1
        return narch

    def add_cell(self, narch):
        narch.append({"filters": self.envelopecell})
        # {"Branch0": "3x3", "Branch1": "3x3sep", "Branch2": "5x5", "Branch3": "5x5sep"} })
        return narch

    def add_widener(self, narch):
        narch.append({"widener": {}})
        # {"Branch0": "3x3", "Branch1": "3x3sep", "Branch2": "5x5", "Branch3": "5x5sep"} })
        return narch

    def group_by_layer(self, samples):
        stats = {}
        for sample in samples[::-1]:
            source_node = int(re.search(r'.*:source-(\d+)dest-(\d+).*', sample).group(1))
            dest_node = int(re.search(r'.*:source-(\d+)dest-(\d+).*', sample).group(2))
            if dest_node not in stats.keys():
                stats[dest_node] = {}
            if source_node not in stats[dest_node].keys():
                print(sample)
                stats[dest_node][source_node] = float(re.search(r'\[(-?\d+\.\d+)\]', sample).group(1))
        return stats


    def insert_skip(self, narch, nsamples=None, offset_count=None, dense_connect=False):
        new_network = narch['network']
        if "skip" not in self.task_config['envelopenet'] or not self.task_config['envelopenet']['skip']:
            return narch

        if dense_connect == True:
            for layer_id, layer in enumerate(narch['network']):
                if "filters" in layer:
                    new_network[layer_id]["inputs"] = []
                    for connections in range(layer_id - 1, 0, -1):
                        new_network[layer_id]["inputs"].append(connections)
        else:
            threshold = 0.5
            new_connections = []
            scalar_filtered_samples = self.get_samples(nsamples, filter_string='scalar')
            l2norm_filtered_samples = self.get_samples(nsamples, filter_string='l2norm')
            scalar_stats = self.group_by_layer(scalar_filtered_samples)
            l2norm_stats = self.group_by_layer(l2norm_filtered_samples)
            # scalar stats are being used right now
            for layer_id, layer in enumerate(narch['network'][1:], start=1):
                if "filters" in layer:
                    if offset_count[layer_id] != offset_count[layer_id - 1]:
                        # New layer has been added at this position, connect densely
                        new_network[layer_id]["inputs"] = []
                        for connections in range(layer_id - 1, 0, -1):
                            new_network[layer_id]["inputs"].append(connections)
                        new_connections.append(layer_id + 1)
                    else:
                        previous_layer_id = layer_id - offset_count[layer_id]
                        if (previous_layer_id+1) in scalar_stats.keys():
                            number_to_keep = len(scalar_stats[previous_layer_id+1]) - int(threshold * len(scalar_stats[previous_layer_id+1]))
                            print("layer_id = {}, number_to_keep = {}, scalar_stats = {}".format(
                                layer_id, number_to_keep, scalar_stats[previous_layer_id+1]))
                            connections = scalar_stats[previous_layer_id+1]
                            pruned_connections = list(zip(*Counter(connections).most_common(number_to_keep)))[0]

                            updated_pruned_connections = []
                            for connection in pruned_connections:
                                offset = offset_count[connection - 1]
                                while offset_count[offset + connection - 1] != offset:
                                    offset = offset_count[offset + connection - 1]
                                new_index = connection + offset
                                updated_pruned_connections.append(new_index)
                            new_network[layer_id]["inputs"] = updated_pruned_connections+ new_connections

        narch['network'] = new_network
        return narch

    def insert_wideners(self, narch):
        # Insert wideners,
        # Space maxwideners equally with a minimum spacing of self.minwidenerintval
        # Last widenerintval may have less layers than others

        #widenerintval= nfilterlayers//self.maxwideners
        widenerintval = len(narch) // self.maxwideners
        if widenerintval < self.minwidenerintval:
            widenerintval = self.minwidenerintval
        #print("Widener interval = " + str(widenerintval))
        nlayer = 1
        insertindices = []
        for layer in narch:
            #print(str(nlayer))
            # Do not add a widener if it is the last layer
            if nlayer % widenerintval == 0 and nlayer != len(narch):
                insertindices.append(nlayer)
            nlayer += 1
        #print("Inserting wideners: " + str(insertindices))
        idxcnt = 0
        for layeridx in insertindices:
            lidx = layeridx + idxcnt
            # Adjust insertion indices after inserts
            #print("Adding widener" + str(lidx))
            narch.insert(lidx, {"widener": {}})
            idxcnt += 1
        #for layer in narch:
            #print(layer)
        return narch

    def gen_randomnet(self):
        self.arch = {"type":"macro","network":[]}
        for stage in range(self.stages):
            starch = []
            for idx in range(int(self.layers_per_stage[stage])):
                starch.append({"filters": {}})
            self.set_outputs(starch, stage,)
            self.arch["network"] += starch
            if stage != self.stages - 1:
                self.arch["network"] = self.add_widener(self.arch)
        #print(self.arch)
        layer = 0
        for stage in range(self.stages):
            # First add at least one block to each layer, to make sure that no
            # layer has zero blocks
            for idx in range(0, self.layers_per_stage[stage]):
                block = random.randint(0, len(self.blocks) - 1)
                blockname = self.blocks[block]
                self.arch["network"][layer]["filters"]["Branch0"] = blockname
                layer += 1
            # Widener
            layer += 1
        #print(self.arch)

        startlayer = 0
        for stage in range(self.stages):
            for idx in range(
                    0,
                    self.numblocks[stage] -
                    self.layers_per_stage[stage]):
                # Pick a random layer
                rlayer = random.randint(0, self.layers_per_stage[stage] - 1)
                # Pick a random block
                block = random.randint(0, len(self.blocks) - 1)
                blockname = self.blocks[block]
                # Increment branch
                alayer = startlayer + rlayer
#                print(
#                    "Start, r, a" +
#                    str(startlayer) +
#                    ":" +
#                    str(rlayer) +
#                    ":" +
#                    str(layer))
                branch = len(self.arch["network"][alayer]["filters"].keys())
                branchname = "Branch" + str(branch)
                self.arch[alayer]["filters"][branchname] = blockname
            # Widener
            startlayer += (self.layers_per_stage[stage] + 1)
        self.arch["network"] = self.insert_skip(self.arch)
        #print(json.dumps(self.arch, indent=4, sort_keys=True))
        return self.arch

    def gen_envelopenet_bystages(self):
        self.arch = {"type":"macro","network":[]}
        #print("Stages: " + str(self.stages))
        #print("Layerperstage: " + str(self.layers_per_stage))
        for stageidx in range(int(self.stages)):
            #print("Stage: " + str(stageidx))
            stage = []
            for idx1 in range(int(self.layers_per_stage[stageidx])):
                #print("Layer : " + str(idx1))
                # TODO  Move this to an evelopenet gen function
                # TODO: Add skip connections
                stage.append({"filters": self.envelopecell})
            self.set_outputs(stage, stageidx)
            if stageidx != int(self.stages) - 1:
                stage = self.add_widener(stage)
            self.arch["network"] += stage
        self.arch = self.insert_skip(self.arch, dense_connect=True)
        #print(json.dumps(self.arch, indent=4, sort_keys=True))
        return self.arch


if __name__ == '__main__':
    base_dir = args.base_dir
    config = args.config
    task = args.task
    g = Generate(base_dir, config, task)
    g.run()
