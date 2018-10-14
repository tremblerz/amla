import tensorflow as tf
slim = tf.contrib.slim

from stubs.tf import cell_init
from stubs.tf import cell_classification
from stubs.tf import cell_main
from stubs.tf import cell_dag


def calc_l2norm(tensor):
    return tf.norm(tensor, ord=2)

def get_macro_net(inputs, log_stats=False, is_training=True,
                    scope='macro_net',
                    arch=None):
    arch = arch["network"]
    net = inputs
    cellnumber = 1  # Init block is 0
    nets = [inputs]

    channelwidth = int(inputs.shape[3])
    for celltype in arch:
        #print(celltype)
        if 'filters' in celltype:
            if "inputs" in celltype.keys():
                all_inputs = [net]
                input_dim = net.shape
                if celltype["inputs"] == "all":
                    for index, reduced_inputs in enumerate(nets[:-1]):
                        while reduced_inputs.shape[1] != input_dim[1]:
                            reduced_inputs = slim.max_pool2d(
                                reduced_inputs, [2, 2], padding='SAME')
                        diffrentiable_scalar = tf.get_variable(name='{}-{}'.format(index, cellnumber), shape=[1],
                            initializer=tf.initializers.random_normal(mean=0.5, stddev=0.01))
                        reduced_inputs = diffrentiable_scalar * reduced_inputs
                        if log_stats:
                            l2_norm = calc_l2norm(reduced_inputs)
                            reduced_inputs = tf.Print(reduced_inputs, [l2_norm],
                                message="l2norm:source-{}dest-{}:".format(index, cellnumber))
                            reduced_inputs = tf.Print(reduced_inputs, [diffrentiable_scalar],
                                message="scalar:source-{}dest-{}:".format(index, cellnumber))
                        all_inputs.append(reduced_inputs)
                else:
                    for input_conn in celltype["inputs"]:
                        reduced_inputs = nets[input_conn]
                        while reduced_inputs.shape[1] != input_dim[1]:
                            reduced_inputs = slim.max_pool2d(
                                reduced_inputs, [2, 2], padding='SAME')
                        diffrentiable_scalar = tf.get_variable(name='{}-{}'.format(input_conn, cellnumber), shape=[1],
                            initializer=tf.initializers.random_normal(mean=0.5, stddev=0.01))
                        reduced_inputs = diffrentiable_scalar * reduced_inputs
                        if log_stats:
                            l2_norm = calc_l2norm(reduced_inputs)
                            reduced_inputs = tf.Print(reduced_inputs, [l2_norm],
                                message="l2norm:source-{}dest-{}:".format(input_conn, cellnumber))
                            reduced_inputs = tf.Print(reduced_inputs, [diffrentiable_scalar],
                                message="scalar:source-{}dest-{}:".format(input_conn, cellnumber))
                        all_inputs.append(reduced_inputs)
                net = tf.concat(axis=3, values=all_inputs)
                num_channels = int(input_dim[3])
                net = slim.conv2d(
                    net, num_channels, [
                        1, 1], scope='BottleneckLayer_1x1_Envelope_' + str(cellnumber))
            outputs = int(celltype["outputs"] /
                          len(celltype["filters"].keys()))
            envelope = cell_main.CellEnvelope(
                cellnumber,
                channelwidth,
                net,
                filters=celltype["filters"],
                log_stats=log_stats,
                outputs=outputs)
            net = envelope.cell(
                net, channelwidth, is_training, filters=celltype["filters"])

        #TODO: Move to stubs
        elif 'widener' in celltype:
            nscope = 'Widener_' + str(cellnumber) + '_MaxPool_2x2'
            net1 = slim.max_pool2d(
                net, [2, 2], scope=nscope, padding='SAME')
            nscope = 'Widener_' + str(cellnumber) + '_conv_3x3'
            net2 = slim.conv2d(
                net, channelwidth, [
                    3, 3], stride=2, scope=nscope, padding='SAME')
            net = tf.concat(axis=3, values=[net1, net2])
            channelwidth *= 2
        elif 'widener2' in celltype:
            for input_conn in celltype["inputs"]:
                reduced_inputs = nets[input_conn]
                while(reduced_inputs.shape[1] != input_dim[1]):
                    reduced_inputs = slim.max_pool2d(reduced_inputs, [2,2], padding='SAME')
                all_inputs.append(reduced_inputs)
            net = tf.concat(axis=3, values=all_inputs)
            num_channels = int(input_dim[3])
            nscope='Widener_'+str(cellnumber)+'_MaxPool_2x2'
            print("Initial #channels={}, after skip={}".format(num_channels, int(net.shape[3])))
            net = slim.max_pool2d(net, [2,2], scope=nscope, padding='SAME')
            channelwidth *= 2
        elif 'auxiliary' in celltype:
            output_units = celltype["outputs"]
            with tf.variable_scope("aux_logits-{}".format(cellnumber)):
                pool = tf.reduce_mean(net, axis=(1,2))
                flatten = slim.flatten(pool)
                fc = slim.fully_connected(flatten, output_units)
                tf.add_to_collection("auxiliary_loss", fc)
            continue
        #elif 'outputs' in celltype:
        #    pass
        else:
            print("Error: Invalid cell definition" + str(celltype))
            exit(-1)

        nets.append(net)
        cellnumber += 1
    return net

def get_micro_net(inputs, log_stats=False, is_training=True,
                    scope='micro_net',
                    arch=None):

    with tf.variable_scope(scope):
        net = cell_dag.get_arch_from_dag(inputs, arch, is_training)
    return net

def gen_network(inputs, log_stats=False, is_training=True,
                    scope='Nacnet',
                    arch=None):

    if arch["type"] == "macro":
        return get_macro_net(inputs, log_stats, is_training,
                    'macro_net', arch)
    elif arch["type"] == "micro":
        return get_micro_net(inputs, log_stats, is_training,
                    'micro_net', arch)
    else:
        print("Invalid architecture specification.")
        exit()


def add_init(inputs, arch, is_training):
    init = cell_init.Init(0)
    net = init.cell(inputs, arch, is_training)
    return net

def add_net(net, log_stats, is_training,
                scope,
                arch):

    net = gen_network(net, log_stats, is_training,
                           scope,
                           arch)
    return net

def add_classification(net, arch, is_training):
    classification = cell_classification.Classification()
    logits = classification.cell(net, arch, is_training)
    return logits


def gen_amlanet(
            inputs,
            arch=None,
            archname=None,
            initcell=None,
            classificationcell=None,
            log_stats=False,
            is_training=True,
            scope='Amlanet'):
    net = add_init(inputs, initcell, is_training)
    end_points = {}
    net = add_net(net, log_stats, is_training,
                               scope,
                               arch
                              )
    linear_softmax = add_classification(
        net, classificationcell, is_training)

    summaries_dir = './summaries/'
    logs_path = summaries_dir + "/" + archname
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # return logits, end_points
    return linear_softmax

