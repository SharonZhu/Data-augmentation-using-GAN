import sys
import os
import tensorflow as tf
# import CycleGAN.utils as utils
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Reader():
  def __init__(self, tfrecords_file, image_size=None,
    min_queue_examples=1000, batch_size=1, num_threads=8, name=''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.image_size = image_size
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()
    self.name = name

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      # reader = tf.TFRecordReader()

      _, serialized_example = self.reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'data_raw': tf.FixedLenFeature([], tf.string)
          })

      # image = tf.decode_raw(features['data_raw'], tf.float32)
      # image = tf.reshape(image, [self.image_size, self.image_size, 3])
      # print(image.shape)
      data = tf.decode_raw(features['data_raw'], tf.float32)
      data = tf.reshape(data, [3])
      print(data.shape)

      datas = tf.train.shuffle_batch(
            [data], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )

      # tf.summary.image('input', datas)
    return datas

def test_reader():
  TRAIN_FILE_1 = '/Users/zhuxinyue/ML/tfrecords/emotion.tfrecords'
  TRAIN_FILE_2 = '/Users/zhuxinyue/ML/tfrecords/faces.tfrecords'

  with tf.Graph().as_default():
    reader1 = Reader(TRAIN_FILE_1, batch_size=2)
    reader2 = Reader(TRAIN_FILE_2, batch_size=2)
    images_op1 = reader1.feed()
    images_op2 = reader2.feed()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        batch_images1, batch_images2 = sess.run([images_op1, images_op2])
        print("image1 shape: {}".format(batch_images1.shape))
        print("image1 shape: {}".format(batch_images2.shape))
        print("="*10)
        step += 1
        sys.exit()
    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

# if __name__ == '__main__':
#   test_reader()
