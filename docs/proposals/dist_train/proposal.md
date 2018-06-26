# Distributed training

### Idea:
 1. #### To run a network across different nodes in a cluster to reduce training time.
 2. #### Allow distributed training for all 3 different modes - train, evaluate and generate.

### Initial Design:
 1. #### [system.json](https://github.com/CiscoAI/amla/blob/master/configs/system.json) should contain cluster specification describing worker nodes and ps nodes. Each node listing should also specify the device to be used. That way, we can leverage multi-gpu training as well by specifying two different gpus as two different worker nodes.
 2. #### For each module i.e. `train`,`evaluate` and `generate`, we will have separate ps and worker nodes listed for all three. Probably, specification would something like this -
         ```
         {
          "cluster": {
                  "scheduler": {"ps": ["host:port", "host:port", ...], "worker":["host:port", ...], "master_id":1},
                  "train": {"ps": ["host:port", ...], "worker":["host:port", ...], "master_id":0},
                  "evaluate": {"ps":["host:port", ...], "worker":["host:port", ...], "master_id":5}
                },
         }
         ```
 3. #### Training style would be asynchronous training with graph replicated across all worker nodes. (Work can be done later to have better distributed training scheme which is more suitable for AutoML algorithms)
