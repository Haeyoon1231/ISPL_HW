import tensorflow as tf
import numpy as np
import glob
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """

    feature = {'encoded': _bytes_feature(img),
               'label': _int64_feature(lab)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """

    train_dir = glob.glob(imagedir + '/train/*')
    test_dir = glob.glob(imagedir + '/test/*')

    # Train dataset
    for i in range(len(train_dir)):
        train_files = glob.glob(train_dir[i] + '/*')
        for j in range(len(train_files)):
            train_file = train_files[j]
            out_dir = train_file[-9:-4] + '.tfrecord'

            img_data = _image_as_bytes(train_file)
            lab = int(train_file[-11])
            example = make_example(img_data, lab)

            # Validation dataset
            if j % 5 == 0:
                writer_val = tf.io.TFRecordWriter(datadir + '/valid_tfrecord/' + out_dir)
                writer_val.write(example)
                writer_val.close()
            # Train dataset
            else:
                writer_train = tf.io.TFRecordWriter(datadir + '/train_tfrecord/' + out_dir)
                writer_train.write(example)
                writer_train.close()

    # Test dataset
    for k in range(len(test_dir)):
        test_files = glob.glob(test_dir[k] + '/*')
        for m in range(len(test_files)):
            test_file = test_files[m]
            writer_test = tf.io.TFRecordWriter(datadir + '/test_tfrecord/' + test_file[-9:-4] + '.tfrecord')
            img_data = _image_as_bytes(test_file)
            lab = int(test_file[-11])
            test_example = make_example(img_data, lab)
            writer_test.write(test_example)
            writer_test.close()


def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """

    filenames = glob.glob(folder + '/*.tfrecord')
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=' '),
                      'label': tf.FixedLenFeature([], tf.int64, default_value=0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = img / 255
    lab = tf.cast(features['label'], tf.int32)
    lab = tf.one_hot(lab, 10)

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch, capacity=batch * 2, num_threads=1, min_after_dequeue=10)

    return img, lab