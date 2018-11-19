import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from hyperopt import hp

import loader
# from train_specgan import _WINDOW_LEN
from specgan import SpecGANDiscriminator
from wavegan import WaveGANDiscriminator

import pprint

"""
  Constants
"""
_FS = 16000
_WINDOW_LEN = 16384
_D_Z = 100
_CLIP_NSTD = 3.
_LOG_EPS = 1e-6

class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final variables'''

    def __init__(self, filename):
        '''hook constructor

        Args:
            filename: where the model will be saved
            variables: the variables that will be saved'''

        self.filename = filename
        # self.variables = variables

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver()

    def end(self, session):
        '''this will be run at session closing'''
        self._saver.save(session, self.filename, global_step=tf.train.get_or_create_global_step())

def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op


def find_data_size(fps, exclude_class):
  # Find out the size of the data
  dummy_it = loader.get_batch(fps, 1, _WINDOW_LEN, False, repeat=False, initializable=True,
                              labels=True, exclude_class=exclude_class)
  dummy_x, _ = dummy_it.get_next()
  train_dataset_size = 0

  with tf.device('cpu:0'):
    with tf.Session() as sess:
      sess.run(dummy_it.initializer)
      try:
        while True:
          sess.run(dummy_x)
          train_dataset_size += 1
      except tf.errors.OutOfRangeError:
        pass

  return train_dataset_size


batch_size_choice = [64, 128]
layers_transferred_choice = [0, 1, 2, 3, 4]
learning_rate_base_Adam_choice = [0.00001, 0.00003, 0.00005, 0.0001, 0.0003, 0.0005]
learning_rate_mode_choice = ['constant', 'decaying']

# Super ugly hack to choose layer_1_size > layer_2_size, and to be able to convert best (returned by hyperopt)
# back to a json
case1_layer1_choice = [2 ** x for x in range(3, 11)]
case2_layer1_choice = [2 ** x for x in range(7, 11)]
case2_layer2_choice = [2 ** x for x in range(3, 7)]

layers_choice = [
    ('case 0', (hp.choice('case0_layer1', range(1))), hp.choice('case0_layer2', range(1))),  # no decision layer at all
    ('case 1', (hp.choice('case1_layer1', case1_layer1_choice)), hp.choice('case1_layer2', range(1))),  # one decision layer
    ('case 2', (hp.choice('case2_layer1', case2_layer1_choice)), hp.choice('case2_layer2', case2_layer2_choice)),  # two decision layers
]

def define_search_space():
    space = {
        'batch_size': hp.choice('batch_size', batch_size_choice),
        'optimizer': hp.choice('optimizer', [
            # {'SGD': {'learning_rate_base': hp.choice('learning_rate_base_SGD', [0.001, 0.003, 0.005])}},
            # {'Adam': {'learning_rate_base': hp.choice('learning_rate_base_Adam', [0.00001, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1])}}
            {'Adam': {'learning_rate_base': hp.choice('learning_rate_base_Adam',
                                                      learning_rate_base_Adam_choice)}}
        ]),

        'learning_rate_mode': hp.choice('learning_rate_mode', learning_rate_mode_choice),
        'layers': hp.choice('layers', layers_choice),
        'layers_transferred': hp.choice('layers_transferred', layers_transferred_choice)
    }

    return space

def transform_hyperopt_result_to_dict_again(hyperopt_param):

    params = {}
    params['batch_size'] = batch_size_choice[hyperopt_param['batch_size']]
    params['layers_transferred'] = layers_transferred_choice[hyperopt_param['layers_transferred']]
    params['optimizer'] = {}
    params['optimizer']['Adam'] ={}
    params['optimizer']['Adam']['learning_rate_base'] = learning_rate_base_Adam_choice[hyperopt_param['learning_rate_base_Adam']]
    params['learning_rate_mode'] = learning_rate_mode_choice[hyperopt_param['learning_rate_mode']]

    layers_case = hyperopt_param['layers']
    if layers_case == 0:
        layer_1_size, layer_2_size = 0, 0
    elif layers_case == 1:
        layer_1_size = case1_layer1_choice[hyperopt_param['case1_layer1']]
        layer_2_size = 0
    elif layers_case == 2:
        layer_1_size = case2_layer1_choice[hyperopt_param['case2_layer1']]
        layer_2_size = case2_layer2_choice[hyperopt_param['case2_layer2']]
    else:
        raise ValueError("Wrong value for layers_case: " + str(layers_case))

    params['layers'] = (layers_case, layer_1_size, layer_2_size)

    return params


def get_cnn_model(params, x, processing_specgan=True):
    batch_size = params['batch_size']
    layers_transferred = params['layers_transferred']
    layer_1_size = params['layers'][1]
    layer_2_size = params['layers'][2]

    with tf.variable_scope('D'):
        D_x_training = SpecGANDiscriminator(x) if processing_specgan else WaveGANDiscriminator(x)  # leave the other parameters default

        last_conv_op_name = 'D_x/D/Maximum_' + str(layers_transferred)
        last_conv_tensor = tf.get_default_graph().get_operation_by_name(last_conv_op_name).values()[0]
        last_conv_tensor = tf.reshape(last_conv_tensor, [batch_size, -1], name="reshape_of_last_transferred_layer")  # Flatten

    with tf.variable_scope('decision_layers'):
        if layer_1_size == 0:
            output = last_conv_tensor
        else:
            with tf.variable_scope('decision_layer_1'):
                output = tf.layers.dense(last_conv_tensor, layer_1_size)

            if layer_2_size != 0:
                with tf.variable_scope('decision_layer_2'):
                    output = tf.layers.dense(output, layer_2_size)

        with tf.variable_scope('cnn_output_layer'):
            output = tf.layers.dense(output, 10)

    return output


def get_optimizer(params, D_loss):

    optimizer = params['optimizer']
    learning_rate_mode = params['learning_rate_mode']

    # Define optmizer
    d_trainer = None
    d_decision_trainer = None
    d_decision_vars = tf.trainable_variables(scope='decision_layers')
    global_step = tf.train.get_or_create_global_step()
    if "SGD" in optimizer:
        learning_rate_base = optimizer['SGD']['learning_rate_base']
        # Define learning rate
        if learning_rate_mode == "decaying":
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
        elif learning_rate_mode == "constant":
            learning_rate = learning_rate_base
        d_trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                            global_step=global_step)
        d_decision_trainer = d_trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                            global_step=global_step,
                                                                                            var_list=d_decision_vars)


    elif "Adam" in optimizer:
        learning_rate_base = optimizer['Adam']['learning_rate_base']
        # Define learning rate
        if learning_rate_mode == "decaying":
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
        elif learning_rate_mode == "constant":
            learning_rate = learning_rate_base
        d_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                 global_step=global_step)
        d_decision_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                 global_step=global_step,
                                                                                 var_list=d_decision_vars)
    else:
        logging.error("Cannot find optimizer: %s", optimizer)

    return d_trainer, d_decision_trainer

