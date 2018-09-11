from stubs.tf import cell_init
from stubs.tf import cell_classification

def get_init_cell(inputs, is_training=False):
    init_block = cell_init.Init()
    #TODO: Load this from JSON
    arch = {
    "Layer0": {"Branch0": {"block": "cutout", "size": 16}},
    "Layer1": {"Branch0": {"block": "conv2d", "kernel_size": [3, 3], "outputs": 112}}
    }
    cell = init_block.cell(inputs, arch, is_training)
    return cell
def get_classification_cell(inputs, is_training=False):
    #TODO: Fix this hardcoded number of 100 in cell constructor
    classification_block = cell_classification.Classification(100)
    arch = {
    "Layer0": {"Branch0": {"block": "reduce_mean", "size": [1, 2]}},
    "Layer1": {"Branch0": {"block": "flatten", "size": [3, 3]}},
    "Layer2": {"Branch0": {"block": "dropout", "keep_prob": 0.7}},
    "Layer3": {"Branch0": {"block": "fc-final", "inputs": 192, "outputs": 10}}
    }
    cell = classification_block.cell(inputs, arch, is_training)
    return cell