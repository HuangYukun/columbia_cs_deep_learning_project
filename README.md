
### ECBM4040 Project
### Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks

Download SVHN dataset from: http://ufldl.stanford.edu/housenumbers/ format 1, or follow the instructions:

```
mkdir data
cd data/
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
wget http://ufldl.stanford.edu/housenumbers/test.tar.gz
wget http://ufldl.stanford.edu/housenumbers/extra.tar.gz
tar -zxf train.tar.gz
tar -zxf test.tar.gz
tar -zxf extra.tar.gz
```

### Please use Python 2.7 to run the program as some of the libraries are defined in Python 2.7
Also, you may need to install the following package:
```
pip install h5py
```


```python
import os
import numpy as np
import h5py
import random
import json
from PIL import Image
import tensorflow as tf
from datetime import datetime
import time
```


```python
# read data, convert data into TF records

tf.app.flags.DEFINE_string('data_dir', './data',
                           'Directory to SVHN (format 1) folders and write the converted files')
FLAGS = tf.app.flags.FLAGS


class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _get_attrs(digit_struct_mat_file, index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file
        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr.value[i].item()].value[0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
            attrs[key] = values
        return attrs

    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self, digit_struct_mat_file):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._example_pointer += 1

        attrs = ExampleReader._get_attrs(digit_struct_mat_file, index)
        label_of_digits = attrs['label']
        length = len(label_of_digits)
        if length > 5:
            # skip this example
            return self.read_and_convert(digit_struct_mat_file)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'length': ExampleReader._int64_feature(length),
            'digits': tf.train.Feature(int64_list=tf.train.Int64List(value=digits))
        }))
        return example


def convert_to_tfrecords(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                         path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)
        print '%d files found in %s' % (total_files, path_to_dataset_dir)

        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            example_reader = ExampleReader(path_to_image_files)
            for index, path_to_image_file in enumerate(path_to_image_files):
                print '(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file)

                example = example_reader.read_and_convert(digit_struct_mat_file)
                if example is None:
                    break

                idx = choose_writer_callback(path_to_tfrecords_files)
                writers[idx].write(example.SerializeToString())
                num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_train_dir = os.path.join(FLAGS.data_dir, 'train')
    path_to_test_dir = os.path.join(FLAGS.data_dir, 'test')
    path_to_extra_dir = os.path.join(FLAGS.data_dir, 'extra')
    path_to_train_digit_struct_mat_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
    path_to_test_digit_struct_mat_file = os.path.join(path_to_test_dir, 'digitStruct.mat')
    path_to_extra_digit_struct_mat_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    [num_train_examples, num_val_examples] = convert_to_tfrecords([(path_to_train_dir, path_to_train_digit_struct_mat_file),
                                                                   (path_to_extra_dir, path_to_extra_digit_struct_mat_file)],
                                                                  [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                  lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    [num_test_examples] = convert_to_tfrecords([(path_to_test_dir, path_to_test_digit_struct_mat_file)],
                                               [path_to_test_tfrecords_file],
                                               lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples,
                               path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':
    tf.app.run(main=main)
```


```python
# deep Convolutional Neural Network model1, one more layer deeper than model2, slightly modified from the paper

class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout
            
        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        with tf.variable_scope('hidden9'):
            conv = tf.layers.conv2d(hidden8, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden9 = dropout

        flatten = tf.reshape(hidden9, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('hidden12'):
            dense = tf.layers.dense(hidden10, units=3072, activation=tf.nn.relu)
            hidden11 = dense

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden11, units=11)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden11, units=11)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden11, units=11)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden11, units=11)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden11, units=11)
            digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss
```


```python
# deep Convolutional Neural Network model2, precisely implemented as the paper states

class Model2(object):

    @staticmethod
    def inference(x, drop_rate):
        with tf.variable_scope('hidden1'):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.contrib.layers.maxout(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout

        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout

        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout

        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout

        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout

        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout

        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout

        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout

        flatten = tf.reshape(hidden8, [-1, 4 * 4 * 192])

        with tf.variable_scope('hidden9'):
            dense = tf.layers.dense(flatten, units=3072, activation=tf.nn.relu)
            hidden9 = dense

        with tf.variable_scope('hidden10'):
            dense = tf.layers.dense(hidden9, units=3072, activation=tf.nn.relu)
            hidden10 = dense

        with tf.variable_scope('digit_length'):
            dense = tf.layers.dense(hidden10, units=7)
            length = dense

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, units=11)
            digit1 = dense

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, units=11)
            digit2 = dense

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, units=11)
            digit3 = dense

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, units=11)
            digit4 = dense

        with tf.variable_scope('digit5'):
            dense = tf.layers.dense(hidden10, units=11)
            digit5 = dense

        length_logits, digits_logits = length, tf.stack([digit1, digit2, digit3, digit4, digit5], axis=1)
        return length_logits, digits_logits

    @staticmethod
    def loss(length_logits, digits_logits, length_labels, digits_labels):
        length_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=length_labels, logits=length_logits))
        digit1_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 0], logits=digits_logits[:, 0, :]))
        digit2_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 1], logits=digits_logits[:, 1, :]))
        digit3_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 2], logits=digits_logits[:, 2, :]))
        digit4_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 3], logits=digits_logits[:, 3, :]))
        digit5_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=digits_labels[:, 4], logits=digits_logits[:, 4, :]))
        loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
        return loss

```


```python
class Donkey(object):
    @staticmethod
    def _preprocess(image):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.multiply(tf.subtract(image, 0.5), 2)
        image = tf.reshape(image, [64, 64, 3])
        image = tf.random_crop(image, [54, 54, 3])
        return image

    @staticmethod
    def _read_and_decode(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'length': tf.FixedLenFeature([], tf.int64),
                'digits': tf.FixedLenFeature([5], tf.int64)
            })

        image = Donkey._preprocess(tf.decode_raw(features['image'], tf.uint8))
        length = tf.cast(features['length'], tf.int32)
        digits = tf.cast(features['digits'], tf.int32)
        return image, length, digits

    @staticmethod
    def build_batch(path_to_tfrecords_file, num_examples, batch_size, shuffled):
        assert tf.gfile.Exists(path_to_tfrecords_file), '%s not found' % path_to_tfrecords_file

        filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], num_epochs=None)
        image, length, digits = Donkey._read_and_decode(filename_queue)

        min_queue_examples = int(0.4 * num_examples)
        if shuffled:
            image_batch, length_batch, digits_batch = tf.train.shuffle_batch([image, length, digits],
                                                                             batch_size=batch_size,
                                                                             num_threads=2,
                                                                             capacity=min_queue_examples + 3 * batch_size,
                                                                             min_after_dequeue=min_queue_examples)
        else:
            image_batch, length_batch, digits_batch = tf.train.batch([image, length, digits],
                                                                     batch_size=batch_size,
                                                                     num_threads=2,
                                                                     capacity=min_queue_examples + 3 * batch_size)
        return image_batch, length_batch, digits_batch

```


```python
class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)

    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 128
        num_batches = num_examples / batch_size
        needs_include_length = False

        with tf.Graph().as_default():
            image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_tfrecords_file,
                                                                         num_examples=num_examples,
                                                                         batch_size=batch_size,
                                                                         shuffled=False)
            length_logits, digits_logits = Model.inference(image_batch, drop_rate=0.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)

            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions

            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)

            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels_string,
                predictions=predictions_string
            )

            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in xrange(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)

        return accuracy_val
```


```python
class Meta(object):
    def __init__(self):
        self.num_train_examples = None
        self.num_val_examples = None
        self.num_test_examples = None

    def save(self, path_to_json_file):
        with open(path_to_json_file, 'w') as f:
            content = {
                'num_examples': {
                    'train': self.num_train_examples,
                    'val': self.num_val_examples,
                    'test': self.num_test_examples
                }
            }
            json.dump(content, f)

    def load(self, path_to_json_file):
        with open(path_to_json_file, 'r') as f:
            content = json.load(f)
            self.num_train_examples = content['num_examples']['train']
            self.num_val_examples = content['num_examples']['val']
            self.num_test_examples = content['num_examples']['test']

```


```python
# train the model

tf.app.flags.DEFINE_string('train_data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 1, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
FLAGS = tf.app.flags.FLAGS


def _train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    with tf.Graph().as_default():
        image_batch, length_batch, digits_batch = Donkey.build_batch(path_to_train_tfrecords_file,
                                                                     num_examples=num_train_examples,
                                                                     batch_size=batch_size,
                                                                     shuffled=True)
        length_logtis, digits_logits = Model.inference(image_batch, drop_rate=0.2)
        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print 'Model restored from file: %s' % path_to_restore_checkpoint_file

            print 'Start training'
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print '=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec)

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print '=> Evaluating on validation dataset...'
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print '=> Model saved to file: %s' % path_to_checkpoint_file
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print '=> patience = %d' % patience
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print 'Finished'


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.train_data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.train_data_dir, 'val.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.train_data_dir, 'meta.json')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    training_options = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _train(path_to_train_tfrecords_file, meta.num_train_examples,
           path_to_val_tfrecords_file, meta.num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file,
           training_options)


if __name__ == '__main__':
    tf.app.run(main=main)
```


```python
# 
tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('checkpoint_dir', './logs/train', 'Directory to read checkpoint files')
tf.app.flags.DEFINE_string('eval_logdir', './logs/eval', 'Directory to write evaluation logs')
FLAGS = tf.app.flags.FLAGS


def _eval(path_to_checkpoint_dir, path_to_eval_tfrecords_file, num_eval_examples, path_to_eval_log_dir):
    evaluator = Evaluator(path_to_eval_log_dir)

    checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue

        accuracy = evaluator.evaluate(path_to_checkpoint, path_to_eval_tfrecords_file, num_eval_examples,
                                      global_step_val)
        print 'Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint, path_to_eval_tfrecords_file, accuracy)


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'meta.json')
    path_to_checkpoint_dir = FLAGS.checkpoint_dir

    path_to_train_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'train')
    path_to_val_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'val')
    path_to_test_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'test')

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _eval(path_to_checkpoint_dir, path_to_train_tfrecords_file, meta.num_train_examples, path_to_train_eval_log_dir)
    _eval(path_to_checkpoint_dir, path_to_val_tfrecords_file, meta.num_val_examples, path_to_val_eval_log_dir)
    _eval(path_to_checkpoint_dir, path_to_test_tfrecords_file, meta.num_test_examples, path_to_test_eval_log_dir)


if __name__ == '__main__':
    tf.app.run(main=main)
```

Tensorboard visualization:
```
    $ tensorboard --logdir ./logs
```

Clean before rerun:
```
    $ rm -rf ./logs
```
