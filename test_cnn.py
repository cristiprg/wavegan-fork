import pickle
import tensorflow as tf

import glob, pprint, json, os
import loader
from specgan import SpecGANDiscriminator
from wavegan import WaveGANDiscriminator
from train_specgan import t_to_f, _WINDOW_LEN, _FS
from train_cnns import get_optimizer

from tensorflow.python.platform import tf_logging as logging
import numpy as np
logging.set_verbosity(tf.logging.INFO)

MAX_EPOCHS = 3

processing_specgan = True

def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op



def get_cnn_model(params, x):
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


def test_model(params):
    batch_size = params['batch_size']

    training_fps = glob.glob(os.path.join(args.data_dir, "train") + '*.tfrecord')
    validation_fps = glob.glob(os.path.join(args.data_dir, "valid") + '*.tfrecord')
    test_fps = glob.glob(os.path.join(args.data_dir, "test") + '*.tfrecord')
    training_fps += validation_fps

    with tf.name_scope('loader'):
        training_dataset = loader.get_batch(training_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, return_dataset=True)
        test_dataset = loader.get_batch(test_fps, batch_size, _WINDOW_LEN, labels=True, repeat=False, return_dataset=True)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        training_init_op = iterator.make_initializer(training_dataset)
        test_init_op = iterator.make_initializer(test_dataset)
        x_wav, labels = iterator.get_next()

        x = t_to_f(x_wav, args.data_moments_mean, args.data_moments_std) if processing_specgan else x_wav

    with tf.name_scope('D_x'):
        cnn_output_logits = get_cnn_model(params, x)

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
    latest_model_ckpt_fp = tf.train.latest_checkpoint(args.load_model_dir) if args.load_model_dir is not None else None

    # saver = tf.train.Saver()
    global_step_op = tf.train.get_or_create_global_step()
    latest_ckpt_fp = tf.train.latest_checkpoint(args.checkpoints_dir) if args.checkpoints_dir is not None else None
    logging.info("Creating session ... ")
    with tf.Session() as sess:

        if args.tensorboard_dir is not None:
            writer = tf.summary.FileWriter(args.tensorboard_dir, sess.graph)

        sess.run(tf.initialize_all_variables())

        if latest_ckpt_fp is not None:
            saver.restore(sess, latest_ckpt_fp)
        elif latest_model_ckpt_fp is not None:
            load_model_saver.restore(sess, latest_model_ckpt_fp)


        for current_epoch in range(MAX_EPOCHS):
            sess.run(training_init_op)
            logging.info("Running epoch " + str(current_epoch))
            try:
                while True:
                    sess.run(cnn_trainer)
            except tf.errors.OutOfRangeError:
                pass

        # Save model
        if args.checkpoints_dir is not None:
            save_path = saver.save(sess, os.path.join(args.checkpoints_dir, "model.ckpt"), global_step=sess.run(global_step_op))
            print("Model saved in path: %s" % save_path)

        logging.info("Testing 1")
        sess.run([acc_reset_op, test_init_op])
        try:
            while True:
                sess.run(acc_update_op)
        except tf.errors.OutOfRangeError:
            current_accuracy = sess.run(acc_op)
            logging.info("Current accuracy = " + str(current_accuracy))


        sess.run(test_init_op)

        logging.info("Testing 2")
        numpy_predictions, numpy_labels = np.array([]), np.array([])
        try:
            while True:
                tmp_pred, tmp_labels = sess.run([predictions, labels])

                numpy_predictions = np.append(numpy_predictions, tmp_pred)
                numpy_labels = np.append(numpy_labels, tmp_labels)
        except tf.errors.OutOfRangeError:
            # End of dataset
            # current_accuracy = sess.run(acc_op)
            # logging.info("accuracy = " + str(current_accuracy))

            print(sess.run(tf.confusion_matrix(numpy_labels, numpy_predictions)))
            with open(os.path.join(".", "predictions.pkl"), 'wb') as f:
                pickle.dump((numpy_labels, numpy_predictions), f)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,help='Data directory')
    parser.add_argument('--tensorboard_dir', type=str, help='Tensorboard directory')
    parser.add_argument('--data_moments_fp', type=str, help='Dataset moments')
    parser.add_argument('--load_model_dir', type=str, help='Directory from where to load the model')
    parser.add_argument('--checkpoints_dir', type=str, help='Directory where to save the checkpoints')

    parser.set_defaults(
        data_dir="/mnt/raid/data/ni/dnn/cristiprg/sc09/",
        tensorboard_dir=None,
        load_model_dir=None,
        checkpoints_dir=None,
        # data_dir="/mnt/scratch/cristiprg/sc09/",
        data_moments_fp="/mnt/antares_raid/home/cristiprg/tmp/pycharm_project_wavegan/train_specgan/moments.pkl")

    args = parser.parse_args()

    if args.checkpoints_dir is not None and args.load_model_dir is not None:
        print("ACHTUNG: both args.checkpoints_dir and args.load_model_dir are set.")
        print("Unless args.checkpoints_dir is empty, the model from args.load_model_dir will be loaded, "
              "otherwise it is ignored.")
        print("Pay attention to logs output by Saver.restore().")

    with open(args.data_moments_fp, 'rb') as f:
        _mean, _std = pickle.load(f, encoding='latin1')

    setattr(args, 'data_moments_mean', _mean)
    setattr(args, 'data_moments_std', _std)

    # Define the architecture
    if processing_specgan:
        s = '{ "batch_size": 128, ' \
                    '"layers_transferred": 3, ' \
                    '"layer_2": "False", ' \
                    '"learning_rate_mode": "decaying",\
                    "optimizer": {"Adam": {"learning_rate_base": 0.001}}}'
        s = json.loads(s)
        s['layer_2'] = False
        pprint.pprint(s)
    else:
        s = '{ "batch_size": 64, ' \
            '"layers_transferred": 4, ' \
            '"layer_2": "False", ' \
            '"learning_rate_mode": "constant",\
            "optimizer": {"Adam": {"learning_rate_base": 0.0001}}}'
        s = json.loads(s)
        s['layer_2'] = False
        pprint.pprint(s)


    # Test:
    test_model(s)