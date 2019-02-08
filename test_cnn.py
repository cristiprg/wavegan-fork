import pickle
import tensorflow as tf

import glob, pprint, json, os
import loader
from specgan import SpecGANDiscriminator
from wavegan import WaveGANDiscriminator
from train_specgan import t_to_f, _WINDOW_LEN, _FS

from util import *

from tensorflow.python.platform import tf_logging as logging
import numpy as np
logging.set_verbosity(tf.logging.INFO)

try:
    from functools import reduce
except ImportError:
    pass

MAX_EPOCHS = 10
processing_specgan = False
mean_delta_accuracy_threshold = .01   # Average of last 3 delta accuracies has to be >= 1
fine_tuning = False

def test_model(params, training_fps, test_fps, args, processing_specgan=processing_specgan, MAX_EPOCHS=MAX_EPOCHS, fine_tuning=False, predictions_pickle=None):
    batch_size = params['batch_size']

    if not hasattr(args, 'checkpoint_iter'):
        setattr(args, 'checkpoint_iter', None)
    if not hasattr(args, 'load_model_dir'):
        setattr(args, 'load_model_dir', None)
    if not hasattr(args, 'checkpoints_dir'):
        setattr(args, 'checkpoints_dir', None)
    if not hasattr(args, 'load_generator_dir'):
        setattr(args, 'load_generator_dir', None)

    logging.info("Testing configuration %s", params)

    with tf.Graph().as_default() as g:

        with tf.name_scope('loader'):
            training_dataset = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, return_dataset=True)
            test_dataset = loader.get_batch(test_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, return_dataset=True)

            train_dataset_size = find_data_size(training_fps, exclude_class=None)
            logging.info("Training datasize = " + str(train_dataset_size))

            test_dataset_size = find_data_size(test_fps, exclude_class=None)
            logging.info("Test datasize = " + str(test_dataset_size))

            iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
            training_init_op = iterator.make_initializer(training_dataset)
            test_init_op = iterator.make_initializer(test_dataset)
            x_wav, labels = iterator.get_next()

            x = t_to_f(x_wav, args.data_moments_mean, args.data_moments_std) if processing_specgan else x_wav

        with tf.name_scope('D_x'):
            cnn_output_logits = get_cnn_model(params, x, processing_specgan=processing_specgan)

        D_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D') + \
                 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decision_layers')

        # Print D summary
        logging.info('-' * 80)
        logging.info('Discriminator vars')
        nparams = 0
        for v in D_vars:
            v_shape = v.get_shape().as_list()
            v_n = reduce(lambda x, y: x * y, v_shape)
            nparams += v_n
            logging.info('{} ({}): {}'.format(v.get_shape().as_list(), v_n, v.name))
        logging.info('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
        logging.info('-' * 80)

        # Define loss and optimizers
        cnn_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cnn_output_logits, labels=tf.one_hot(labels, 10))
        cnn_trainer, d_decision_trainer = get_optimizer(params, cnn_loss)

        predictions = tf.argmax(cnn_output_logits, axis=1)

        # Define accuracy for validation
        acc_op, acc_update_op, acc_reset_op = resettable_metric(tf.metrics.accuracy, 'foo',
                                                                labels=labels,
                                                                predictions=tf.argmax(cnn_output_logits, axis=1))

        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D') +
                                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decision_layers'))

        load_model_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='D'))
        if args.checkpoint_iter is not None:
            latest_model_ckpt_fp = tf.train.get_checkpoint_state(checkpoint_dir=args.train_dir).all_model_checkpoint_paths[
                args.checkpoint_iter]
        else:
            latest_model_ckpt_fp = tf.train.latest_checkpoint(args.load_model_dir) if args.load_model_dir is not None else None

        # saver = tf.train.Saver()
        global_step_op = tf.train.get_or_create_global_step()
        latest_ckpt_fp = tf.train.latest_checkpoint(args.checkpoints_dir) if args.checkpoints_dir is not None else None

        def get_accuracy():
            sess.run([acc_reset_op, test_init_op])
            try:
                while True:
                    sess.run(acc_update_op)
            except tf.errors.OutOfRangeError:
                current_accuracy = sess.run(acc_op)
            return current_accuracy

        def get_mean_delta_accuracy(accuracy):
            accuracies_feature_extraction.append(accuracy)
            # Early stopping?
            if len(accuracies_feature_extraction) >= 4:
                mean_delta_accuracy = (accuracies_feature_extraction[-1] - accuracies_feature_extraction[-4]) * 1.0 / 3
                return mean_delta_accuracy
            return 99999

        def run_trainer(trainer, message_prefix):
            for current_epoch in range(MAX_EPOCHS):
                sess.run(training_init_op)
                try:
                    while True:
                        sess.run(trainer)
                except tf.errors.OutOfRangeError:
                    if current_epoch < MAX_EPOCHS / 2.0:  # early stopping, but when?
                        logging.info("%s finished epoch %d" % (message_prefix, current_epoch))
                        continue
                    accuracy = get_accuracy()
                    logging.info("%s epoch %d accuracy = %f" % (message_prefix, current_epoch, accuracy))
                    mean_delta_accuracy = get_mean_delta_accuracy(accuracy)

                    if mean_delta_accuracy < mean_delta_accuracy_threshold:
                        logging.info("Early stopping, mean_delta_accuracy = " + str(mean_delta_accuracy))
                        break


        logging.info("Creating session ... ")
        with tf.Session(graph=g) as sess:

            if args.tensorboard_dir is not None:
                writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)

            sess.run(tf.initialize_all_variables())

            if latest_ckpt_fp is not None:
                saver.restore(sess, latest_ckpt_fp)
            elif latest_model_ckpt_fp is not None:
                load_model_saver.restore(sess, latest_model_ckpt_fp)

            # Feature extraction
            accuracies_feature_extraction = []
            run_trainer(d_decision_trainer, "Feature extraction")

            # Fine tuning
            if fine_tuning is True:
                accuracies_feature_extraction = []
                run_trainer(cnn_trainer, "Fine tuning")

            # Save model
            if args.checkpoints_dir is not None:
                save_path = saver.save(sess, os.path.join(args.checkpoints_dir, "model.ckpt"), global_step=sess.run(global_step_op))
                print("Model saved in path: %s" % save_path)

            fine_tuning_accuracy = get_accuracy()
            logging.info("Fine tuning accuracy = " + str(fine_tuning_accuracy))


            sess.run(test_init_op)

            if predictions_pickle is not None:
                logging.info("Testing 2")
                numpy_predictions, numpy_labels = np.array([]), np.array([])
                try:
                    while True:
                        tmp_pred, tmp_labels = sess.run([predictions, labels])

                        numpy_predictions = np.append(numpy_predictions, tmp_pred)
                        numpy_labels = np.append(numpy_labels, tmp_labels)
                except tf.errors.OutOfRangeError:

                    logging.info("Pickling predictions to " + predictions_pickle)

                    with open(predictions_pickle, 'wb') as f:
                        pickle.dump((numpy_labels, numpy_predictions), f)

        return fine_tuning_accuracy




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,help='Data directory')
    parser.add_argument('--tensorboard_dir', type=str, help='Tensorboard directory')
    parser.add_argument('--data_moments_fp', type=str, help='Dataset moments')
    parser.add_argument('--load_model_dir', type=str, help='Directory from where to load the model')
    parser.add_argument('--checkpoints_dir', type=str, help='Directory where to save the checkpoints')
    parser.add_argument('--train_data_percentage', type=int,
                           help='Take the first train_data_percentage % of the data')
    parser.add_argument('--architecture', type=str, help='The architecture to use in JSON format')
    parser.add_argument('--gan_type', type=str, help='Either specgan or wavegan')
    parser.add_argument('--fine_tuning', dest='fine_tuning', default=False, action='store_true')

    parser.set_defaults(
        data_dir="/mnt/raid/data/ni/dnn/cristiprg/sc09/",
        tensorboard_dir=None,
        load_model_dir=None,
        checkpoints_dir=None,
        train_data_percentage=100,
        architecture=None,
        # data_dir="/mnt/scratch/cristiprg/sc09/",
        data_moments_fp="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/train_specgan/moments.pkl",
        gan_type=None)

    args = parser.parse_args()

    if args.gan_type is None or (args.gan_type != "specgan" and args.gan_type != "wavegan"):
        print("ERROR: --gan_type not correctly set!")
        exit(1)

    processing_specgan = True if args.gan_type == "specgan" else False

    if args.checkpoints_dir is not None and args.load_model_dir is not None:
        print("ACHTUNG: both args.checkpoints_dir and args.load_model_dir are set.")
        print("Unless args.checkpoints_dir is empty, the model from args.load_model_dir will be loaded, "
              "otherwise it is ignored.")
        print("Pay attention to logs output by Saver.restore().")

    with open(args.data_moments_fp, 'rb') as f:
        try:
            _mean, _std = pickle.load(f)
        except UnicodeDecodeError:
            _mean, _std = pickle.load(f, encoding='latin1')
    setattr(args, 'data_moments_mean', _mean)
    setattr(args, 'data_moments_std', _std)

    # Define the architecture
    if args.architecture is None:
        if processing_specgan:
            s = '{ "batch_size": 64, "layers_transferred": 4, "layers": [0, 0, 0], "learning_rate_mode": "constant", "optimizer": {"Adam": {"learning_rate_base": 0.0001}}}'
            s = json.loads(s)
            # s['layer_2'] = False
        else:
            s = '{ "batch_size": 64, ' \
                '"layers_transferred": 4, ' \
                '"layers": [0, 256, 16], ' \
                '"learning_rate_mode": "constant",\
                "optimizer": {"Adam": {"learning_rate_base": 0.0001}}}'
            s = json.loads(s)
            s['layer_2'] = False
    else:
        print(args.architecture)
        s = json.loads(args.architecture)
    # pprint.pprint(s)

    # Test:
    training_fps = glob.glob(os.path.join(args.data_dir,
                                 "train") + '*.tfrecord') \
                   + glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
    training_fps = sorted(training_fps)
    length = len(training_fps)
    # training_fps = training_fps[:(int(args.train_data_percentage / 100.0 * length))]  # keep the beginning
    training_fps = training_fps[(int((100 - args.train_data_percentage) / 100.0 * length)):]  # keep the ending
    test_fps = glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord')


    # args.load_model_dir = args.load_model_dir[:-1] if args.load_model_dir[-1] == '/' else args.load_model_dir
    # pickle_name = args.load_model_dir + "_performance.pkl"

    # Change for computing populations/distribution of total accuracies (not labels) for rand sampling
    # training_fps += test_fps
    for seed in range(10):
        # training_fps, test_fps = loader.split_files_test_val(training_fps, train_size=0.9, seed=seed)
        # logging.info("len(training_fps) = %d" % len(training_fps))
        # logging.info("len(test_fps) = %d", len(test_fps))
        acc = test_model(s, training_fps=training_fps, test_fps=test_fps, args=args, processing_specgan=processing_specgan,
                   MAX_EPOCHS=MAX_EPOCHS, fine_tuning=args.fine_tuning, predictions_pickle=None)

        logging.info("Test accuracy: %s %s %s %s", str(s), args.train_data_percentage, seed,  acc)
