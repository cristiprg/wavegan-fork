import multiprocessing

import tensorflow as tf
from hyperopt import hp, fmin, STATUS_OK, STATUS_FAIL, Trials, tpe
import hyperopt.pyll.stochastic

import sys
import os
import cPickle as pickle
from pprint import pprint

import loader
from specgan import SpecGANDiscriminator
from train_specgan import t_to_f, _WINDOW_LEN, _FS

import pprint

from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(tf.logging.INFO)

DRY_RUN = True  # Whether or not to run just one training iteration (as oposed to multiple epochs)
MAX_EPOCHS = 20
mean_delta_accuracy_threshold = .5   # Average of last 3 delta accuracies has to be >= 1

# /mnt/raid/ni/dnn/install_cuda-9.0/../build/cudnn-9.0-v7/lib64:/usr/local/cuda-9.0/lib64:/mnt/raid/ni/dnn/install_cuda-9.0/lib:/mnt/raid/ni/dnn/install_cuda-9.0/../build/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/mnt/raid/ni/dnn/install_cuda-9.0/lib:/mnt/raid/ni/dnn/install_cuda-8.0/../build/cudnn-8.0/lib64:/usr/local/cuda-8.0/lib64:/mnt/raid/ni/dnn/install_cuda-8.0/lib

def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op


def define_search_space():
    space = {
        'batch_size': hp.choice('batch_size', [64, 128]),
        'optimizer': hp.choice('optimizer', [
            # {'SGD': {'learning_rate_base': hp.choice('learning_rate_base_SGD', [0.001, 0.003, 0.005])}},
            {'Adam': {'learning_rate_base': hp.choice('learning_rate_base_Adam', [0.00001, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1])}}
        ]),

        'learning_rate_mode': hp.choice('learning_rate_mode', ['constant', 'decaying']),

        'layer_1_size': hp.choice('layer_1_size', [2 ** x for x in range(4, 8)]),

        'layer_2': hp.choice('layer_2', [
            False,
            {
                'layer_2_size': hp.choice('layer_2_size', [2 ** x for x in range(4, 8)])
            }
        ]),

        'layers_transferred': hp.choice('layers_transferred', [1, 2, 3, 4])
    }

    return space


def get_cnn_model(params, x, validation=False):
    batch_size = params['batch_size']
    layers_transferred = params['layers_transferred']
    with tf.variable_scope('D', reuse=validation):
        D_x_training = SpecGANDiscriminator(x)  # leave the other parameters default

        last_conv_op_name = 'D_x_'+str('validation' if validation else 'training')+'/D/Maximum_' + str(layers_transferred)
        last_conv_tensor = tf.get_default_graph().get_operation_by_name(last_conv_op_name).values()[0]
        last_conv_tensor = tf.reshape(last_conv_tensor, [batch_size, -1])  # Flatten

    with tf.variable_scope('decision_layers', reuse=validation):
        cnn_decision_layer = tf.layers.dense(last_conv_tensor, 10)

    return cnn_decision_layer


def get_optimizer(params, D_loss):

    optimizer = params['optimizer']
    learning_rate_mode = params['learning_rate_mode']

    # Define optmizer
    d_trainer = None
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

    elif "Adam" in optimizer:
        learning_rate_base = optimizer['Adam']['learning_rate_base']
        # Define learning rate
        if learning_rate_mode == "decaying":
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
        elif learning_rate_mode == "constant":
            learning_rate = learning_rate_base
        d_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                 global_step=global_step)
    else:
        logging.error("Cannot find optimizer: %s", optimizer)

    return d_trainer


def evaluate_model_process(params, q):

    logging.info("Running configuration %s", params)

    batch_size = params['batch_size']

    train_dataset_size = 18620

    # Define input training, validation and test data
    logging.info("Preparing data ... ")
    training_fps = glob.glob(os.path.join(args.data_dir, "train") + '*.tfrecord')
    validation_fps = glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
    test_fps = glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord')

    with tf.name_scope('loader'):
        # Training is the only one that repeats
        training_iterator = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, initializable=True)
        x_wav_training, labels_training = training_iterator.get_next()
        x_training = t_to_f(x_wav_training, args.data_moments_mean, args.data_moments_std)

        validation_iterator = loader.get_batch(validation_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, initializable=True)
        x_wav_validation, labels_validation = validation_iterator.get_next()
        x_validation = t_to_f(x_wav_validation, args.data_moments_mean, args.data_moments_std)

    # Get the discriminator and put extra layers
    with tf.name_scope('D_x_training'):
        cnn_training = get_cnn_model(params, x_training, validation=False)

    with tf.name_scope('D_x_validation'):
        cnn_validation = get_cnn_model(params, x_validation, validation=True)

    # Define loss and optimizers
    cnn_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_training, labels=tf.one_hot(labels_training, 10))
    cnn_trainer = get_optimizer(params, cnn_loss)

    # Define accuracy for validation
    acc_op, acc_update_op, acc_reset_op = resettable_metric(tf.metrics.accuracy, 'foo',
                                                            labels=labels_validation,
                                                            predictions=tf.argmax(cnn_validation, axis=1))

    # Restore the variables of the discriminator if necessary
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D'))
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)

    logging.info("Creating session ... ")
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=None,
            log_step_count_steps=10,  # don't save checkpoints, not worth for parameter tuning
            save_checkpoint_secs=None) as sess:

        # Main training loop
        status = STATUS_OK
        logging.info("Entering main loop ...")
        accuracies = []

        logdir = "./tensorboard_development/"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        for current_epoch in range(MAX_EPOCHS):
            sess.run(training_iterator.initializer)
            try:
                while True:
                    sess.run(cnn_trainer)
            except tf.errors.OutOfRangeError:
                # End of training dataset
                pass

            # Validation
            sess.run([acc_reset_op, validation_iterator.initializer])
            try:
                while True:
                    sess.run(acc_update_op)
            except tf.errors.OutOfRangeError:
                    # End of dataset
                    current_accuracy = sess.run(acc_op)
                    logging.info("epoch" + str(current_epoch) + " accuracy = " + str(current_accuracy))
                    accuracies.append(current_accuracy)

            # Early stopping?
            if len(accuracies) >= 4:
                mean_delta_accuracy = (accuracies[-1] - accuracies[-4]) * 1.0 / 3

                if mean_delta_accuracy < mean_delta_accuracy_threshold:
                    logging.info("Early stopping, mean_delta_accuracy = " + str(mean_delta_accuracy))
                    break

            if current_epoch >= MAX_EPOCHS:
                logging.info("Stopping after " + str(MAX_EPOCHS) + " epochs!")

        q.put({
            'loss': -accuracies[-1],  # last accuracy; also, return the negative for maximization problem (accuracy)
            'status': status,
            # -- store other results like this
            'accuracy_history': accuracies

        })

        logging.info("Result: %s %s", params, str(accuracies[-1]))


def evaluate_model(params):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=evaluate_model_process, args=(params,q))
    p.start()
    res = q.get()
    p.join()
    return res


def evaluate_model_backup(params):
    batch_size = params['batch_size']
    layers_transferred = params['layers_transferred']
    optimizer = params['optimizer']
    learning_rate_mode = params['learning_rate_mode']

    dataset_size = 18620

    # Define input training, validation and test data
    training_fps = glob.glob(os.path.join(args.data_dir, "train") + '*.tfrecord')
    validation_fps = glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
    test_fps = glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord')

    with tf.name_scope('loader'):
        # Training is the only one that repeats
        x_wav_training, labels_training = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=True)
        x_training = t_to_f(x_wav_training, args.data_moments_mean, args.data_moments_std)

        x_wav_validation, labels_validation = loader.get_batch(validation_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False)
        x_validation = t_to_f(x_wav_validation, args.data_moments_mean, args.data_moments_std)

        # x_wav_test, labels_test = loader.get_batch(test_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False)
        # x_test = t_to_f(x_wav_test, args.data_moments_mean, args.data_moments_std)


    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x = SpecGANDiscriminator(x_training)  # leave the other parameters default
    # with tf.name_scope('D_x_validation'), tf.variable_scope('D', reuse=True):
    #     D_x_validation = SpecGANDiscriminator(x_validation)

    # The last layer is a function of 'layers_transferred'
    last_conv_op_name = 'D_x/D/downconv_'+str(layers_transferred)+'/conv2d/Conv2D'
    last_conv_tensor = tf.get_default_graph().get_operation_by_name(last_conv_op_name).values()[0]
    last_conv_tensor = tf.reshape(last_conv_tensor, [batch_size, -1])  # Flatten

    with tf.variable_scope('decision_layers'):
        D_x = tf.layers.dense(last_conv_tensor, 10, name='decision_layer', reuse=False)
    # with tf.variable_scope('decision_layers', reuse=True):
    #     D_x_validation = tf.layers.dense(last_conv_tensor, 10, name='decision_layer', reuse=True)

    decision_layers_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decision_layers')


    # Define cross-entropy loss for the SC09 dataset
    # labels = tf.map_fn(to_int, labels, dtype=tf.int32)
    # TODO: read labels directly from data source, not placeholders. This forces a copy from GPU's memory to CPU's memory.
    labels_placeholder = tf.placeholder(tf.int32, shape=[None], name='labels_placeholder')
    D_loss = tf.nn.softmax_cross_entropy_with_logits(logits=D_x, labels=tf.one_hot(labels_training, 10))

    # Define optmizer
    d_trainer = None
    global_step = tf.Variable(0, trainable=False)
    if optimizer == "SGD":
        learning_rate_base = params['learning_rate_base_SGD']

        # Define learning rate
        if learning_rate_mode == "decaying":
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
        elif learning_rate_mode == "constant":
            learning_rate = learning_rate_base
        d_trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                            global_step=tf.train.get_or_create_global_step())

    elif optimizer == "Adam":
        learning_rate_base = params['learning_rate_base_Adam']

        # Define learning rate
        if learning_rate_mode == "decaying":
            learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
        elif learning_rate_mode == "constant":
            learning_rate = learning_rate_base
        d_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,
                                                                                 global_step=tf.train.get_or_create_global_step())

    # Load model saved on disk - either specgan or wavegan
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D'))
    latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
    print "latest_ckpt_fp = ", latest_ckpt_fp

    # Get some summaries
    x_wav_summary_op = tf.summary.audio('x_wav', x_wav_training, _FS, max_outputs=batch_size)
    logdir = "./tensorboard_development/"

    # validation_y_placeholder = tf.placeholder(tf.int32, [None])
    # acc, acc_op = tf.metrics.accuracy(labels=labels_validation, predictions=tf.argmax(D_x_validation, axis=1))

    checkpoint_dir = os.path.join(args.train_dir, "development", "checkpoints")
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=checkpoint_dir,
            log_step_count_steps=100,
            save_checkpoint_secs=600) as sess:

        # Restore model saved by SPECGAN if there's nothing saved in the checkpoint_dir
        # (i.e. if this training has crashed and is now restarted)
        # if tf.train.latest_checkpoint(checkpoint_dir) is None:
        #     saver.restore(sess, latest_ckpt_fp)

        # TODO: check daca valorile din D_x si D_x_validation sunt aceleasi, i.e. ii acelasi model

        # Main training loop
        current_epoch = 0

        writer = tf.summary.FileWriter(logdir, sess.graph)
        writer.flush()

        return

        while not sess.should_stop():
            _, current_step = sess.run([d_trainer, tf.train.get_or_create_global_step()])
            current_epoch_test = current_step / dataset_size
            if current_epoch_test == current_epoch + 1:  # New epoch hook

                # Validation
                while True:
                    try:
                        current_accuracy = sess.run(acc_op)
                    except tf.errors.OutOfRangeError:
                        # End of dataset
                        print "current accuracy = ", current_accuracy
                        break



if __name__ == '__main__':
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                           help='Data directory')

    parser.add_argument('--train_dir', type=str,
                        help='Training directory')

    parser.add_argument('--data_moments_fp', type=str,
                           help='Dataset moments')

    parser.set_defaults(
        train_dir="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/train_specgan/",
        data_dir="/mnt/raid/data/ni/dnn/cristiprg/sc09/",
        data_moments_fp="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/train_specgan/moments.pkl"
    )

    args = parser.parse_args()

    with open(args.data_moments_fp, 'rb') as f:
      _mean, _std = pickle.load(f)
    setattr(args, 'data_moments_mean', _mean)
    setattr(args, 'data_moments_std', _std)

    # latest_ckpt_fp = tf.train.latest_checkpoint(args.data_dir)

    trials = Trials()
    best = fmin(fn=evaluate_model, space=define_search_space(),
                algo=tpe.suggest, max_evals=100, trials=trials)

    with open("hyperopt.txt", "w") as fout:
        fout.write("Best = \n")
        pprint.pprint(best, fout)
        fout.write('Best accuracy = %s\n' % str(best['result']['loss']))

        fout.write("All trials = \n")
        for trial in trials.trials:
            pprint.pprint(trial, fout)


    # params = {}
    # params['batch_size'] = 64
    # params['layers_transferred'] = 2
    # params['optimizer'] = 'SGD'
    # params['learning_rate_base_SGD'] = 0.1
    # params['learning_rate_mode'] = 'constant'
    # evaluate_model(params)

