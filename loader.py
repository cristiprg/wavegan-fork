import tensorflow as tf


def f0(): return tf.constant('0', dtype=tf.string)


def f1(): return tf.constant('1', dtype=tf.string)


def f2(): return tf.constant('2', dtype=tf.string)


def f3(): return tf.constant('3', dtype=tf.string)


def f4(): return tf.constant('4', dtype=tf.string)


def f5(): return tf.constant('5', dtype=tf.string)


def f6(): return tf.constant('6', dtype=tf.string)


def f7(): return tf.constant('7', dtype=tf.string)


def f8(): return tf.constant('8', dtype=tf.string)


def f9(): return tf.constant('9', dtype=tf.string)


def f_error(): return tf.constant('-1', dtype=tf.string)  # TODO: raise error?


def single_label_to_int(label):
  pred_fn_pairs = {
    tf.equal(label, "Zero"): f0,
    tf.equal(label, "One"): f1,
    tf.equal(label, "Two"): f2,
    tf.equal(label, "Three"): f3,
    tf.equal(label, "Four"): f4,
    tf.equal(label, "Five"): f5,
    tf.equal(label, "Six"): f6,
    tf.equal(label, "Seven"): f7,
    tf.equal(label, "Eight"): f8,
    tf.equal(label, "Nine"): f9,
  }
  return tf.string_to_number(tf.case(pred_fn_pairs, default=f_error, exclusive=True), out_type=tf.int32)


"""
  Data loader
  fps: List of tfrecords
  batch_size: Resultant batch size
  window_len: Size of slice to take from each example
  first_window: If true, always take the first window in the example, otherwise take a random window
  repeat: If false, only iterate through dataset once
  labels: If true, return (x, y), else return x
  buffer_size: Number of examples to queue up (larger = more random)
"""
def get_batch(
    fps,
    batch_size,
    window_len,
    first_window=False,
    repeat=True,
    labels=False,
    buffer_size=8192,
    initializable=False):
  def _mapper(example_proto):
    features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
    if labels:
      features['label'] = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

    example = tf.parse_single_example(example_proto, features)
    wav = example['samples']
    if labels:
      label = tf.reduce_join(example['label'], 0)
      label = single_label_to_int(label)

    if first_window:
      # Use first window
      wav = wav[:window_len]
    else:
      # Select random window
      wav_len = tf.shape(wav)[0]

      start_max = wav_len - window_len
      start_max = tf.maximum(start_max, 0)

      start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

      wav = wav[start:start+window_len]

    wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]])

    wav.set_shape([window_len, 1])

    if labels:
      return wav, label
    else:
      return wav

  dataset = tf.data.TFRecordDataset(fps)
  dataset = dataset.map(_mapper)
  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  if repeat:
    dataset = dataset.repeat()

  # dataset.prefetch(1)
  if not initializable:
    iterator = dataset.make_one_shot_iterator()
  else:
    return dataset.make_initializable_iterator()


  return iterator.get_next()
