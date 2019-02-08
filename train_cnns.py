# import multiprocessing
import queue
import time
import tensorflow as tf
from hyperopt import hp, fmin, STATUS_OK, STATUS_FAIL, Trials, tpe
import hyperopt.pyll.stochastic

import glob
import parser
import sys

PYTHON_VERSION = 2
if sys.version_info[0] == 3:
    PYTHON_VERSION = 3

import datetime
import os
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from pprint import pprint
import json
import math

import loader
from train_specgan import t_to_f, _WINDOW_LEN, _FS
from util import *
from test_cnn import test_model as test_model_with_params

import pprint

from tensorflow.python.platform import tf_logging as logging
logging.set_verbosity(tf.logging.INFO)


DRY_RUN = True  # Whether or not to run just one training iteration (as oposed to multiple epochs)
MAX_EPOCHS = 10
train_dataset_size = None
mean_delta_accuracy_threshold = .01   # Average of last 3 delta accuracies has to be >= 1
processing_specgan = False
perform_feature_extraction = True
perform_fine_tuning = False
record_hyperopt_feature_extraction = True
perform_hyperopt = True
perform_test_best_hyperopt = True
# skip_training_percentage = 0
global_train_data_percentage = None
global_checkpoint_iter = None
# Don't forget to set up the proper interpreter

args = None  # TODO: make this variable visible in subproce ss
evaluated_configs = {}
# /mnt/raid/ni/dnn/install_cuda-9.0/../build/cudnn-9.0-v7/lib64:/usr/local/cuda-9.0/lib64:/mnt/raid/ni/dnn/install_cuda-9.0/lib:/mnt/raid/ni/dnn/install_cuda-9.0/../build/cudnn-9.0-v7.1/lib64:/usr/local/cuda-9.0/lib64:/mnt/raid/ni/dnn/install_cuda-9.0/lib:/mnt/raid/ni/dnn/install_cuda-8.0/../build/cudnn-8.0/lib64:/usr/local/cuda-8.0/lib64:/mnt/raid/ni/dnn/install_cuda-8.0/lib


def evaluate_model_process(params, seed, q):

    global args

    max_tries = 10
    for current_try in range(max_tries):  # try 10 times
        with tf.Graph().as_default() as g:

            batch_size = params['batch_size']

            # Define input training, validation and test data
            # TODO: think of a deterministic way to do the data split, GAN + Classifier train + valid. Valid out of Train must be seed-able
            logging.info("Preparing data ... ")
            # training_fps = glob.glob(os.path.join(args.data_dir, "train") + '*.tfrecord')# + glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
            training_fps = args.training_fps
            training_fps, validation_fps = loader.split_files_test_val(training_fps, train_size=0.9, seed=seed)

            with tf.name_scope('loader'):
                training_dataset = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False,
                                                    return_dataset=True)
                validation_dataset = loader.get_batch(validation_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False,
                                                      return_dataset=True)

                train_dataset_size = find_data_size(training_fps, exclude_class=None)
                logging.info("Training datasize = " + str(train_dataset_size))

                valid_dataset_size = find_data_size(validation_fps, exclude_class=None)
                logging.info("Validation datasize = " + str(valid_dataset_size))

                iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
                training_init_op = iterator.make_initializer(training_dataset)
                validation_init_op = iterator.make_initializer(validation_dataset)
                x_wav, labels = iterator.get_next()

                x = t_to_f(x_wav, args.data_moments_mean, args.data_moments_std) if processing_specgan else x_wav

            # Get the discriminator and put extra layers
            with tf.name_scope('D_x'):
                cnn_output_logits = get_cnn_model(params, x, processing_specgan=processing_specgan)

            # Define loss and optimizers
            cnn_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_output_logits, labels=tf.one_hot(labels, 10))
            cnn_trainer, d_decision_trainer = get_optimizer(params, cnn_loss)

            # Define accuracy performance measure
            acc_op, acc_update_op, acc_reset_op = resettable_metric(tf.metrics.accuracy, 'foo',
                                                                    labels=labels,
                                                                    predictions=tf.argmax(cnn_output_logits, axis=1))

            # Restore the variables of the discriminator if necessary
            saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D'))
            logging.info("args.train_dir = %s" % args.train_dir)
            if args.train_dir is not None:
                latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
            # ckpt_fp = tf.train.get_checkpoint_state(checkpoint_dir=args.train_dir).all_model_checkpoint_paths[args.checkpoint_iter]

            tensorboard_session_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(
                global_train_data_percentage) + "_" + str(seed)

            def initialize_iterator(sess, skip=False):
                sess.run(training_init_op)

                # if skip:
                #     batches_to_skip = math.ceil(1.0 * train_dataset_size * skip_training_percentage / 100.0 / batch_size)
                #     logging.info("Skipping " + str(batches_to_skip) + " batches.")
                #     for _ in range(batches_to_skip):
                #         sess.run(x)  # equivalent to doing nothing with these training samples

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False
            config.gpu_options.per_process_gpu_memory_fraction = 0.01
            logging.info("Creating session ... ")
            try:
                # with tf.train.MonitoredTrainingSession(
                #         checkpoint_dir=None,
                #         log_step_count_steps=10,  # don't save checkpoints, not worth for parameter tuning
                #         save_checkpoint_secs=None) as sess:
                with tf.Session(config=config, graph=g) as sess:
                    sess.run(tf.global_variables_initializer())
                    # Don't forget to RESTORE!!!
                    # saver.restore(sess, os.path.join(args.train_dir, "model.ckpt"))
                    if args.train_dir is not None:
                        saver.restore(sess, latest_ckpt_fp)

                    # saver.restore(sess, ckpt_fp)

                    # Main training loop
                    status = STATUS_OK
                    logging.info("Entering main loop ...")

                    logdir = "./tensorboard_wavegan_cnn/" + str(tensorboard_session_name) + "/"
                    # writer = tf.summary.FileWriter(logdir, sess.graph)

                    # nr_training_batches = math.ceil(train_dataset_size / batch_size *
                    #                                 train_data_percentage / 100.0 *
                    #                                 (100 - skip_training_percentage) / 100.0)
                    nr_training_batches = train_dataset_size

                    logging.info("Training batches: " + str(nr_training_batches))

                    # Step 1: Train decision layer only
                    if perform_feature_extraction:
                        accuracies_feature_exctraction = []
                        for current_epoch in range(MAX_EPOCHS):
                            # sess.run(training_iterator.initializer)
                            # logging.info("Aici 1")
                            initialize_iterator(sess, skip=True)
                            # logging.info("Aici 2")
                            current_step = -1  # this step is the step within an epoch, therefore different from the global step
                            try:
                                while True:
                                    # logging.info("Aici 3")
                                    sess.run(d_decision_trainer)
                                    # logging.info("Aici 4")

                                    current_step += 1

                                    # Stop training after x% of training data seen
                                    if current_step > nr_training_batches:
                                        break

                            except tf.errors.OutOfRangeError:
                                # End of training dataset
                                pass

                            logging.info("Stopped training at epoch step: " + str(current_step))
                            # Validation
                            sess.run([acc_reset_op, validation_init_op])
                            try:
                                while True:
                                    sess.run(acc_update_op)
                            except tf.errors.OutOfRangeError:
                                    # End of dataset
                                    current_accuracy = sess.run(acc_op)
                                    logging.info("Feature extraction epoch" + str(current_epoch) + " accuracy = " + str(current_accuracy))
                                    accuracies_feature_exctraction.append(current_accuracy)

                            # Early stopping?
                            if len(accuracies_feature_exctraction) >= 4:
                                mean_delta_accuracy = (accuracies_feature_exctraction[-1] - accuracies_feature_exctraction[-4]) * 1.0 / 3

                                if mean_delta_accuracy < mean_delta_accuracy_threshold:
                                    logging.info("Early stopping, mean_delta_accuracy = " + str(mean_delta_accuracy))
                                    break

                            if current_epoch >= MAX_EPOCHS:
                                logging.info("Stopping after " + str(MAX_EPOCHS) + " epochs!")

                        logging.info("Result feature extraction: %s %s %s %s %s", params, global_train_data_percentage, seed, global_checkpoint_iter, str(accuracies_feature_exctraction[-1]))

                    # Step 2: Continue training everything
                    if perform_fine_tuning:
                        accuracies_fine_tuning = []
                        for current_epoch in range(MAX_EPOCHS):
                            # sess.run(training_iterator.initializer)
                            initialize_iterator(sess, skip=True)
                            current_step = -1  # this step is the step within an epoch, therefore different from the global step
                            try:
                                while True:
                                    sess.run(cnn_trainer)

                                    current_step += 1

                                    # Stop training after x% of training data seen
                                    if current_step > nr_training_batches:
                                        break

                            except tf.errors.OutOfRangeError:
                                # End of training dataset
                                pass

                            logging.info("Stopped training at epoch step: " + str(current_step))
                            # Validation
                            sess.run([acc_reset_op, validation_init_op])
                            try:
                                while True:
                                    sess.run(acc_update_op)
                            except tf.errors.OutOfRangeError:
                                # End of dataset
                                current_accuracy = sess.run(acc_op)
                                logging.info("Fine tuning epoch" + str(current_epoch) + " accuracy = " + str(current_accuracy))
                                accuracies_fine_tuning.append(current_accuracy)

                            # Early stopping?
                            if len(accuracies_fine_tuning) >= 4:
                                mean_delta_accuracy = (accuracies_fine_tuning[-1] - accuracies_fine_tuning[-4]) * 1.0 / 3

                                if mean_delta_accuracy < mean_delta_accuracy_threshold:
                                    logging.info("Early stopping, mean_delta_accuracy = " + str(mean_delta_accuracy))
                                    break

                            if current_epoch >= MAX_EPOCHS:
                                logging.info("Stopping after " + str(MAX_EPOCHS) + " epochs!")

                        logging.info("Result fine tuning: %s %s %s %s %s", params, global_train_data_percentage, seed,
                                     global_checkpoint_iter, str(accuracies_fine_tuning[-1]))

                    recorded_accuracy = accuracies_feature_exctraction[-1] if record_hyperopt_feature_extraction is True else accuracies_fine_tuning[-1]
                    q.put({
                        'loss': -recorded_accuracy,  # last accuracy; also, return the negative for maximization problem (accuracy)
                        'status': status,
                        # -- store other results like this
                        # 'accuracy_history': accuracies_fine_tuning

                    })
            except tf.errors.ResourceExhaustedError:
                if current_try == max_tries - 1:
                    logging.info("Got Resources Exhausted Error - Returning FAIL: %s %s %s %s", params, global_train_data_percentage,
                                 seed, global_checkpoint_iter)
                    q.put({
                        'loss': -999,
                        'status': STATUS_FAIL
                    })
                else:
                    logging.info("Got Resources Exhausted Error - Retrying %d: %s %s %s %s", current_try, params,
                                 global_train_data_percentage, seed, global_checkpoint_iter)
            except Exception as e:
                if current_try == max_tries - 1:
                    logging.info("Got Following Exception - Returning FAIL: %s %s %s %s", params,
                                 global_train_data_percentage,
                                 seed, global_checkpoint_iter)
                    logging.info(e)
                    q.put({
                        'loss': -998,
                        'status': STATUS_FAIL
                    })
                else:
                    logging.info("Got Following Exception - Retrying %d: %s %s %s %s", current_try, params,
                                 global_train_data_percentage, seed, global_checkpoint_iter)
                    logging.info(e)
            else:
                # Everything good, success, so don't retry
                break



def evaluate_model_hyperopt(params):
    global evaluated_configs

    key = hash(str(params))
    logging.info("key="+str(key))
    if key in evaluated_configs:
        evaluated_configs[key] += 1
    else:
        evaluated_configs[key] = 0
    seed = evaluated_configs[key]

    logging.info("Running configuration %s, data_size %s, seed %s, checkpoint_iter %s", params, str(global_train_data_percentage),
                 str(seed), str(global_checkpoint_iter))

    # q = multiprocessing.Queue()
    q = queue.Queue()
    evaluate_model_process(params, seed, q)
    # p = multiprocessing.Process(target=evaluate_model_process, args=(params, seed, q))
    # p.daemon = True
    # p.start()
    res = q.get()
    logging.debug("Res = " + str(res))
    # p.join()

    time.sleep(1)
    return res


def evaluate_model(params):
    # q = multiprocessing.Queue()
    #
    # for train_data_percentage in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]:
    #     for seed in range(7):
    #         p = multiprocessing.Process(target=evaluate_model_process, args=(params, train_data_percentage, seed, q))
    #         p.start()
    #         res = q.get()
    #         p.join()
    #
    # return res
    raise NotImplementedError()


def test_model_wrapper(q, params, training_fps, test_fps, args, processing_specgan=processing_specgan, MAX_EPOCHS=MAX_EPOCHS, fine_tuning=perform_fine_tuning, predictions_pickle=None):
    """
    This must already run in another process
    TODO: fix this thing with the parameters
    """
    best_model_accuracy = test_model_with_params(params, training_fps, test_fps, args,
                                                 processing_specgan=processing_specgan, MAX_EPOCHS=MAX_EPOCHS,
                                                 fine_tuning=perform_fine_tuning, predictions_pickle=predictions_pickle)
    q.put(best_model_accuracy)

#
# def evaluate_model_backup(params):
#     batch_size = params['batch_size']
#     layers_transferred = params['layers_transferred']
#     optimizer = params['optimizer']
#     learning_rate_mode = params['learning_rate_mode']
#
#     dataset_size = 18620
#
#     # Define input training, validation and test data
#     training_fps = glob.glob(os.path.join(args.data_dir, "train") + '*.tfrecord')
#     validation_fps = glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
#     test_fps = glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord')
#
#     with tf.name_scope('loader'):
#         # Training is the only one that repeats
#         x_wav_training, labels_training = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=True)
#         x_training = t_to_f(x_wav_training, args.data_moments_mean, args.data_moments_std)
#
#         x_wav_validation, labels_validation = loader.get_batch(validation_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False)
#         x_validation = t_to_f(x_wav_validation, args.data_moments_mean, args.data_moments_std)
#
#         # x_wav_test, labels_test = loader.get_batch(test_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False)
#         # x_test = t_to_f(x_wav_test, args.data_moments_mean, args.data_moments_std)
#
#
#     with tf.name_scope('D_x'), tf.variable_scope('D'):
#         D_x = SpecGANDiscriminator(x_training)  # leave the other parameters default
#     # with tf.name_scope('D_x_validation'), tf.variable_scope('D', reuse=True):
#     #     D_x_validation = SpecGANDiscriminator(x_validation)
#
#     # The last layer is a function of 'layers_transferred'
#     last_conv_op_name = 'D_x/D/downconv_'+str(layers_transferred)+'/conv2d/Conv2D'
#     last_conv_tensor = tf.get_default_graph().get_operation_by_name(last_conv_op_name).values()[0]
#     last_conv_tensor = tf.reshape(last_conv_tensor, [batch_size, -1])  # Flatten
#
#     with tf.variable_scope('decision_layers'):
#         D_x = tf.layers.dense(last_conv_tensor, 10, name='decision_layer', reuse=False)
#     # with tf.variable_scope('decision_layers', reuse=True):
#     #     D_x_validation = tf.layers.dense(last_conv_tensor, 10, name='decision_layer', reuse=True)
#
#     decision_layers_parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decision_layers')
#
#
#     # Define cross-entropy loss for the SC09 dataset
#     # labels = tf.map_fn(to_int, labels, dtype=tf.int32)
#     # TODO: read labels directly from data source, not placeholders. This forces a copy from GPU's memory to CPU's memory.
#     labels_placeholder = tf.placeholder(tf.int32, shape=[None], name='labels_placeholder')
#     D_loss = tf.nn.softmax_cross_entropy_with_logits(logits=D_x, labels=tf.one_hot(labels_training, 10))
#
#     # Define optmizer
#     d_trainer = None
#     global_step = tf.Variable(0, trainable=False)
#     if optimizer == "SGD":
#         learning_rate_base = params['learning_rate_base_SGD']
#
#         # Define learning rate
#         if learning_rate_mode == "decaying":
#             learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
#         elif learning_rate_mode == "constant":
#             learning_rate = learning_rate_base
#         d_trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss,
#                                                                                             global_step=tf.train.get_or_create_global_step())
#
#     elif optimizer == "Adam":
#         learning_rate_base = params['learning_rate_base_Adam']
#
#         # Define learning rate
#         if learning_rate_mode == "decaying":
#             learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 100000, 0.96, staircase=False)
#         elif learning_rate_mode == "constant":
#             learning_rate = learning_rate_base
#         d_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss,
#                                                                                  global_step=tf.train.get_or_create_global_step())
#
#     # Load model saved on disk - either specgan or wavegan
#     saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D'))
#     latest_ckpt_fp = tf.train.latest_checkpoint(args.train_dir)
#     print "latest_ckpt_fp = ", latest_ckpt_fp
#
#     # Get some summaries
#     x_wav_summary_op = tf.summary.audio('x_wav', x_wav_training, _FS, max_outputs=batch_size)
#     logdir = "./tensorboard_development/"
#
#     # validation_y_placeholder = tf.placeholder(tf.int32, [None])
#     # acc, acc_op = tf.metrics.accuracy(labels=labels_validation, predictions=tf.argmax(D_x_validation, axis=1))
#
#     checkpoint_dir = os.path.join(args.train_dir, "development", "checkpoints")
#     with tf.train.MonitoredTrainingSession(
#             checkpoint_dir=checkpoint_dir,
#             log_step_count_steps=100,
#             save_checkpoint_secs=600) as sess:
#
#         # Restore model saved by SPECGAN if there's nothing saved in the checkpoint_dir
#         # (i.e. if this training has crashed and is now restarted)
#         # if tf.train.latest_checkpoint(checkpoint_dir) is None:
#         #     saver.restore(sess, latest_ckpt_fp)
#
#         # TODO: check daca valorile din D_x si D_x_validation sunt aceleasi, i.e. ii acelasi model
#
#         # Main training loop
#         current_epoch = 0
#
#         writer = tf.summary.FileWriter(logdir, sess.graph)
#         writer.flush()
#
#         return
#
#         while not sess.should_stop():
#             _, current_step = sess.run([d_trainer, tf.train.get_or_create_global_step()])
#             current_epoch_test = current_step / dataset_size
#             if current_epoch_test == current_epoch + 1:  # New epoch hook
#
#                 # Validation
#                 while True:
#                     try:
#                         current_accuracy = sess.run(acc_op)
#                     except tf.errors.OutOfRangeError:
#                         # End of dataset
#                         print "current accuracy = ", current_accuracy
#                         break
#


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

    parser.add_argument('--predictions_pickles_directory', type=str, help='Where to store the predictions on the test'
                                                                          'set for the best models selected by hyperopt')

    parser.add_argument('--train_data_percentages', type=str, help='Comma separated list of percentages: eg: 100,90,80')
    parser.add_argument('--checkpoint_iters', type=str, help='Comma separated list of checkpoint indexes to load, default: None - latest')
    parser.add_argument('--discard_first_data_percentage', type=int, help='Discard first x% of training data. Useful for training 90-10 GANs etc.')
    parser.add_argument('--gan_type', type=str, help='Either specgan or wavegan')
    parser.add_argument('--fine_tuning', dest='fine_tuning', default=False, action='store_true')

    if PYTHON_VERSION == 2:
        parser.set_defaults(
            #train_dir="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/" +
            #          ("train_specgan_90" if processing_specgan else "train_wavegan_90"),
            train_dir=None,
            data_dir="/mnt/raid/data/ni/dnn/cristiprg/sc09/",
            # data_dir="/mnt/scratch/cristiprg/sc09/",
            data_moments_fp="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/train_specgan/moments.pkl",
            predictions_pickles_directory="./hyperopt_pickles",
            train_data_percentages="100",
            checkpoint_iters=None,
            discard_first_data_percentage=0,
            gan_type=None,
        )
    else:
        parser.set_defaults(
            train_dir="D:\\Scoala\\Drums\\models\\train_wavegan\\",
            # data_dir="D:\\Scoala\\Drums\\data\\sc09\\",
            data_dir="d:\\Scoala\\Drums\\data\\wave_drums\\drums\\",
            # data_moments_fp="D:\\Scoala\\Drums\\models\\train_specgan\\moments.pkl"
            data_moments_fp="d:\\Scoala\\Drums\\data\\wave_drums\\drums\\moments.pkl"

        )

    args = parser.parse_args()

    if args.fine_tuning is True:
        perform_fine_tuning = True
        record_hyperopt_feature_extraction = False

    if args.gan_type is None or (args.gan_type != "specgan" and args.gan_type != "wavegan"):
        print("ERROR: --gan_type not correctly set!")
        exit(1)

    processing_specgan = True if args.gan_type == "specgan" else False

    with open(args.data_moments_fp, 'rb') as f:
        print("PYTHON_VERSION = " + str(PYTHON_VERSION))
        if PYTHON_VERSION == 2:
            _mean, _std = pickle.load(f)
        else:
            _mean, _std = pickle.load(f, encoding='latin1')
    setattr(args, 'data_moments_mean', _mean)
    setattr(args, 'data_moments_std', _std)

    # latest_ckpt_fp = tf.train.latest_checkpoint(args.data_dir)

    fps = glob.glob(os.path.join(args.data_dir,
                                          "train") + '*.tfrecord') \
          + glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
    fps = sorted(fps)

    # 70 - 30 specific config: discard the first 70% of the data
    if args.discard_first_data_percentage > 0:
        length = len(fps)
        fps = fps[(int(args.discard_first_data_percentage / 100.0 * length)):]

    train_data_percentages = [int(x) for x in args.train_data_percentages.split(",")]
    # checkpoint_iters = [int(x) for x in args.checkpoint_iters.split(",")]
    if perform_hyperopt:
        for global_train_data_percentage in train_data_percentages:
            # for checkpoint_iter in checkpoint_iters:
            #     global_checkpoint_iter = checkpoint_iter
                mean_delta_accuracy_threshold /= 100 / global_train_data_percentage  # allow more iterations for less data?
                # setattr(args, 'checkpoint_iter', checkpoint_iter)
                # logging.info("checkpoint_iter = " + str(checkpoint_iter))

                logging.info("global_train_data_percentage = " + str(global_train_data_percentage))

                length = len(fps)
                training_fps = fps[:(int(global_train_data_percentage / 100.0 * length))]
                # training_fps = fps
                setattr(args, 'training_fps', training_fps)

                trials = Trials()
                evaluated_configs = {}
                best = fmin(fn=evaluate_model_hyperopt, space=define_search_space(),
                            algo=tpe.suggest, max_evals=50, trials=trials)

                setattr(args, 'load_model_dir', args.train_dir)
                # setattr(args, 'checkpoints_dir', None)
                setattr(args, 'tensorboard_dir', None)

                if perform_test_best_hyperopt:
                    for seed in range(10):
                        # pickle_name = str(global_train_data_percentage) + "_" + str(checkpoint_iter) + ".pkl"
                        # pickle_name = str(global_train_data_percentage) + "_rand_init.pkl"
                        # pickle_name = args.train_dir + "_performance_" + str(seed) + ".pkl"
                        params = transform_hyperopt_result_to_dict_again(best)
                        # q = multiprocessing.Queue()
                        q = queue.Queue()

                        # p = multiprocessing.Process(target=test_model_wrapper, args=(q,
                        #     params,
                        #     training_fps,
                        #     glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord'),
                        #     args,
                        #     processing_specgan,
                        #     1,
                        #     os.path.join(args.predictions_pickles_directory, pickle_name)))

                        # Same-process version equivalent
                        test_model_wrapper(q,
                            params=params,
                            training_fps=training_fps,
                            test_fps=glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord'),
                            args=args,
                            processing_specgan=processing_specgan,
                            MAX_EPOCHS=10,
                            fine_tuning=perform_fine_tuning,
                            predictions_pickle=None
                            # predictions_pickle=os.path.join(args.predictions_pickles_directory, pickle_name)
                        )
                        # p.start()
                        acc = q.get()
                        # p.join()
                        logging.info("Test accuracy: %s %s %s %s", str(params), global_train_data_percentage, seed, acc)

        # with open("hyperopt.txt", "w") as fout:
        #     fout.write("Best = \n")
        #     pprint.pprint(best, fout)
        #     fout.write('Best accuracy = %s\n' % str(best['result']['loss']))
        #
        #     fout.write("All trials = \n")
        #     for trial in trials.trials:
        #         pprint.pprint(trial, fout)
    else:
        # params = {}
        # params['batch_size'] = 128
        # params['layers_transferred'] = 3
        # params['optimizer'] = {}
        # params['optimizer']['Adam'] ={}
        # params['optimizer']['Adam']['learning_rate_base'] = 0.001
        # params['layer_1_size'] = 32
        # params['layer_2'] = False
        # params['learning_rate_mode'] = 'decaying'

        s = '{ "batch_size": 64, ' \
            '"layers_transferred": 4, ' \
            '"layer_2": "False", ' \
            '"learning_rate_mode": "constant",\
            "optimizer": {"Adam": {"learning_rate_base": 0.0001}}}'
        s = json.loads(s)
        s['layer_2'] = False
        pprint.pprint(s)

        evaluate_model(s)
