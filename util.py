import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from hyperopt import hp

import loader
from train_specgan import _WINDOW_LEN
from specgan import SpecGANDiscriminator
from wavegan import WaveGANDiscriminator


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
layers_transferred_choice = [1, 2, 3, 4]
learning_rate_base_Adam_choice = [0.00001, 0.00003, 0.00005, 0.0001, 0.0003, 0.0005]
learning_rate_mode_choice = ['constant', 'decaying']
layer_1_size_choice = [2 ** x for x in range(4, 8)]
layer_2_size_choice = [2 ** x for x in range(4, 8)]

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

        'layer_1_size': hp.choice('layer_1_size', layer_1_size_choice),

        'layer_2': hp.choice('layer_2', [
            False,
            {
                'layer_2_size': hp.choice('layer_2_size', layer_2_size_choice)
            }
        ]),

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
    params['layer_1_size'] = layer_1_size_choice[hyperopt_param['layer_1_size']]
    # params['layer_2'] = False if hyperopt_param['layer_2'] == 0 else True
    #
    # if params['layer_2']:
    #     params['layer_2']['layer_2_size'] = layer_2_size_choice[hyperopt_param['layer_2_size']]

    if hyperopt_param['layer_2'] == 1:  # which means True - check with the choice
        params['layer_2'] = {}
        params['layer_2']['layer_2_size'] = layer_2_size_choice[hyperopt_param['layer_2_size']]
    else:
        params['layer_2'] = False

    return params


def get_cnn_model(params, x, processing_specgan=True):
    batch_size = params['batch_size']
    layers_transferred = params['layers_transferred']
    with tf.variable_scope('D'):
        D_x_training = SpecGANDiscriminator(x) if processing_specgan else WaveGANDiscriminator(x)  # leave the other parameters default

        last_conv_op_name = 'D_x/D/Maximum_' + str(layers_transferred)
        last_conv_tensor = tf.get_default_graph().get_operation_by_name(last_conv_op_name).values()[0]
        last_conv_tensor = tf.reshape(last_conv_tensor, [batch_size, -1])  # Flatten

    with tf.variable_scope('decision_layers'):
        cnn_decision_layer = tf.layers.dense(last_conv_tensor, 10)

    return cnn_decision_layer


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

