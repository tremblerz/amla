import tensorflow as tf

def get_arch_from_dag(inputs, arch):
	nets = [inputs]
	for index, cell_type in enumerate(arch["network"]):

		assert cell_type in arch, "{} is not defined in arch".format(cell_type)
		cell = arch[cell_type]
		subgraph = {}

		with tf.variable_scope(cell_type + "-" + str(index)):
			for node in sorted(cell.keys(), key = lambda nodename: int(nodename[4:])):
				with tf.variable_scope(node):

					if type(cell[node]) == int:
						subgraph[node] = nets[cell[node]]
					elif type(cell[node]) == dict:
						target_node = cell[node]["node"]
						inputs = cell[node]["input"]
						normalized_inputs = []
						if len(inputs) > 1:
							min_output_size = subgraph[inputs[0]].shape[1]
							for input_node in inputs:
								min_output_size = min(min_output_size, subgraph[input_node].shape[1])
							for input_node in inputs:
								reduced_input = subgraph[input_node]
								if reduced_input.shape[1] > min_output_size:
									print("Reducing {}".format(input_node))
									with tf.variable_scope("reduction_layers"):
										while reduced_input.shape[1] != min_output_size:
											reduced_input = tf.layers.conv2d(reduced_input, reduced_input.shape[3],
												[3,3], strides=2, padding="SAME")
								normalized_inputs.append(reduced_input)

						#print(subgraph)
						with tf.variable_scope(target_node["type"]):
							# Concat
							if target_node["type"] == "concat":
								net = tf.concat(axis=3, values=normalized_inputs)
								subgraph[node] = net
							# Add
							elif target_node["type"] == "add":
								target_inputs = []
								max_output_filter = subgraph[inputs[0]].shape[3]
								# To have all featuremaps with same channels
								for input_node in normalized_inputs:
									max_output_filter = max(max_output_filter, input_node.shape[3])
								for input_node in normalized_inputs:
									scaled_input = input_node
									if input_node.shape[3] < max_output_filter:
										with tf.variable_scope("bottleneck_layer"):
											scaled_input = tf.layers.conv2d(scaled_input, max_output_filter,
												[1,1], padding="SAME")
									target_inputs.append(scaled_input)
								net = tf.add_n(target_inputs)
								subgraph[node] = net
							# Convolution and pooling
							else:
							    # get parameters
								input_node = subgraph[inputs[0]]
								kernel_size = int(target_node["type"][0]), int(target_node["type"][2])
								conv_filters = target_node.get("filters", input_node.shape[3])
								stride = target_node.get("stride", 1)
								padding = target_node.get("activation", "SAME")
								activation = target_node.get("activation", tf.nn.relu)

								if target_node["type"].endswith("maxpool"):
									net = tf.layers.max_pooling2d(input_node, kernel_size,
										strides=stride, padding=padding)
								elif target_node["type"].endswith("avgpool"):
									net = tf.layers.average_pooling2d(input_node, kernel_size,
										strides=stride, padding=padding)
								elif target_node["type"].endswith("sep"):
									net = tf.layers.separable_conv2d(input_node, conv_filters,
										kernel_size, strides=stride, padding=padding,
										activation=activation)
								else:
									net = tf.layers.conv2d(input_node, conv_filters,
										kernel_size, strides=stride, padding=padding,
										activation=activation)
								subgraph[node] = net
					else:
						print("Invalid node specification. Only Integer and Dict allowed")
			
			out_filters = nets[index].shape[3]
			if cell_type == "reduction_cell":
				out_filters *= 2
			
			with tf.variable_scope("bottleneck_layer"):
				net = tf.layers.conv2d(net, out_filters,
					[1,1], strides=1)
		nets.append(net)
		print(net)
	return net