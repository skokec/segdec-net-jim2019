from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import tensor_shape


class NetInputProcessing(object):
    def __init__(self, batch_size, num_preprocess_threads, input_size = None, mask_size = None, num_readers = 1, input_queue_memory_factor=16, use_random_rotation=False, ensure_posneg_balance=True):
        self.batch_size = batch_size

        self.input_size = input_size  # input_size = (height, width, depth)
        self.mask_size = mask_size  # mask_size = (height, width, depth)

        self.num_preprocess_threads = num_preprocess_threads
        self.num_readers = num_readers
        self.input_queue_memory_factor = input_queue_memory_factor
        self.use_random_rotation = use_random_rotation
        self.ensure_posneg_balance = ensure_posneg_balance

        if self.num_preprocess_threads is None:
            raise Exception("Missing num_preprocess_threads argument")

        if self.num_preprocess_threads % 1:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 1 (%d % 1 != 0).', self.num_preprocess_threads)

        if self.num_readers is None:
            raise Exception("Missing num_readers argument")

        if self.num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

    def add_inputs_nodes(self, dataset, train):
      """Generate batches of ImageNet images for evaluation.
    
      Use this function as the inputs for evaluating a network.
    
      Note that some (minimal) image preprocessing occurs during evaluation
      including central cropping and resizing of the image to fit the network.
    
      Args:
        dataset: instance of Dataset class specifying the input_data.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
          None defaults to FLAGS.num_preprocess_threads.
    
      Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                           image_size, 3].
        labels: 1-D integer Tensor of [FLAGS.batch_size].
      """

      # Force all input processing onto CPU in order to reserve the GPU for
      # the forward inference and back-propagation.
      with tf.device('/cpu:0'):
        images, labels, img_names = self.batch_inputs(dataset, train=train)

      return images, labels, img_names

    def batch_inputs(self, dataset, train):
        """Contruct batches of training or evaluation examples from the image input_data.
  
        Args:
          dataset: instance of Dataset class specifying the input_data.
            See input_data.py for details.
          batch_size: integer
          train: boolean
          num_preprocess_threads: integer, total number of preprocessing threads
          num_readers: integer, number of parallel readers
  
        Returns:
          images: 4-D float Tensor of a batch of images
          labels: 1-D integer Tensor of [batch_size].
  
        Raises:
          ValueError: if data is not found
        """
        with tf.name_scope('batch_processing'):
            data_files = dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this input_data')

            # Create filename_queue
            if train:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=True,
                                                                capacity=16)
            else:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=False,
                                                                capacity=1)

            # Approximate number of examples per shard.
            examples_per_shard = 1024
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = examples_per_shard * self.input_queue_memory_factor
            if train:
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * self.batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
            else:
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * self.batch_size,
                    dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if self.num_readers > 1:
                enqueue_ops = []
                for _ in range(self.num_readers):
                    reader = dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = dataset.reader()
                _, example_serialized = reader.read(filename_queue)


            pos_queue = None
            neg_queue = None

            if self.batch_size < 2:
                pos_queue = tf.RandomShuffleQueue(name="pos-queue", capacity=10, min_after_dequeue=5, dtypes=[tf.float32, tf.float32, tf.string])
                neg_queue = tf.RandomShuffleQueue(name="neg-queue", capacity=10, min_after_dequeue=5, dtypes=[tf.float32, tf.float32, tf.string])

            pos_queue_enq = []
            neg_queue_enq = []

            with tf.name_scope('split-merge'):
                if train and self.ensure_posneg_balance:
                    images_and_masks = []
                    for thread_id in range(self.num_preprocess_threads):
                        # Parse a serialized Example proto to extract the image and metadata.
                        image_buffer, mask_buffer, img_name_ = self.parse_example_proto(example_serialized)

                        image_ = self.image_preprocessing(image_buffer, img_size=(self.input_size[0],self.input_size[1]), num_channels=self.input_size[2])
                        mask_ = self.image_preprocessing(mask_buffer, img_size=(self.mask_size[0],self.mask_size[1]), num_channels=self.mask_size[2])

                        image_ = tf.expand_dims(image_, 0)
                        mask_ = tf.expand_dims(mask_, 0)
                        img_name_ = tf.expand_dims(img_name_,0)

                        img_shape = tf.TensorShape([image_.shape[1], image_.shape[2], image_.shape[3]])
                        mask_shape = tf.TensorShape([mask_.shape[1], mask_.shape[2], mask_.shape[3]])
                        img_name_shape = tf.TensorShape([])

                        # initialize pos/neg queues with proper shape size on first
                        if pos_queue is None or neg_queue is None:
                            pos_queue = tf.RandomShuffleQueue(name="pos-queue", capacity=10, min_after_dequeue=5, dtypes=[tf.float32, tf.float32, tf.string], shapes=[img_shape, mask_shape, img_name_shape])
                            neg_queue = tf.RandomShuffleQueue(name="neg-queue", capacity=10, min_after_dequeue=5, dtypes=[tf.float32, tf.float32, tf.string], shapes=[img_shape, mask_shape, img_name_shape])

                        is_pos = tf.squeeze(tf.reduce_sum(mask_,[1,2], keep_dims=False))

                        neg_mask = tf.less_equal(is_pos, 0)

                        pos_idx = tf.reshape(tf.where([tf.logical_not(neg_mask)]), [-1])
                        neg_idx = tf.reshape(tf.where([neg_mask]),[-1])

                        pos_data = [tf.gather(image_, pos_idx),
                                    tf.gather(mask_, pos_idx),
                                    tf.gather(img_name_, pos_idx)]
                        neg_data = [tf.gather(image_, neg_idx),
                                    tf.gather(mask_, neg_idx),
                                    tf.gather(img_name_, neg_idx)]

                        pos_queue_enq.append(pos_queue.enqueue_many(pos_data))
                        neg_queue_enq.append(neg_queue.enqueue_many(neg_data))


                    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(pos_queue, pos_queue_enq))
                    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(neg_queue, neg_queue_enq))


                    if self.batch_size >= 2:
                        if self.batch_size % 2 != 0:
                            raise Exception("'batch_size' mod 2 != 0 ! only even batch sizes supported at the moment")

                        num_deque = int(self.batch_size / 2)

                        pos_data = pos_queue.dequeue_many(num_deque)
                        neg_data = neg_queue.dequeue_many(num_deque)

                        concat_data = [tf.concat([pos_data[0], neg_data[0]], axis=0, name='Concat-img'),
                                       tf.concat([pos_data[1], neg_data[1]], axis=0, name='Concat-mask'),
                                       tf.concat([pos_data[2], neg_data[2]], axis=0, name='Concat-img-name')]

                        # randomly permute within batch size (is this even necessary ??)
                        idx = tf.Variable(range(0, self.batch_size), trainable=False, dtype=tf.int32)
                        idx = tf.random_shuffle(idx)

                        images = tf.gather(concat_data[0],idx)
                        masks = tf.gather(concat_data[1],idx)
                        img_names = tf.gather(concat_data[2],idx)

                    else:
                        # positive only
                        #images, masks, img_names = pos_queue.dequeue()

                        # negative only
                        #images, masks, img_names = neg_queue.dequeue()

                        # mix 50/50
                        counter = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32)

                        counter = tf.assign_add(counter, 1)
                        condition_term = tf.equal(tf.mod(counter, 2), tf.constant(0))
                        images, masks, img_names  = tf.cond(condition_term,
                                                            lambda: pos_queue.dequeue(),
                                                            lambda: neg_queue.dequeue())

                        if self.use_random_rotation:
                            images.set_shape(tensor_shape.as_shape([None, None, 1]))
                            masks.set_shape(tensor_shape.as_shape([None, None, 1]))

                            # randomly rotate image by 90 degrees
                            rot_factor = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
                            rot_factor = tf.gather(rot_factor,0)

                            images = tf.image.rot90(images, k=rot_factor)
                            masks = tf.image.rot90(masks, k=rot_factor)

                        images = tf.expand_dims(images,axis=0)
                        masks = tf.expand_dims(masks, axis=0)
                        img_names = tf.expand_dims(img_names, axis=0)
                else:

                    # Parse a serialized Example proto to extract the image and metadata.
                    image_buffer, mask_buffer, img_names = self.parse_example_proto(example_serialized)

                    images = self.image_preprocessing(image_buffer,
                                                      img_size=(self.input_size[0], self.input_size[1]),
                                                      num_channels=self.input_size[2])
                    masks = self.image_preprocessing(mask_buffer, img_size=(self.mask_size[0], self.mask_size[1]),
                                                     num_channels=1)



                    images = tf.expand_dims(images, axis=0)
                    masks = tf.expand_dims(masks, axis=0)
                    img_names = tf.expand_dims(img_names, axis=0)

            # Reshape images into these desired dimensions.
            images = tf.cast(images, tf.float32)
            masks = tf.cast(masks, tf.float32)

            images.set_shape(tensor_shape.as_shape([self.batch_size, None, None, self.input_size[2]]))
            masks.set_shape(tensor_shape.as_shape([self.batch_size, self.input_size[0], self.input_size[1], self.mask_size[2]]))

            # Display the training images in the visualizer.
            tf.summary.image('images', images)
            tf.summary.image('masks', masks)

            return images, masks, img_names

    def decode_png(self, image_buffer, num_channels, scope=None):
      """Decode a PNG string into one 3-D float image Tensor.
    
      Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for name_scope.
      Returns:
        3-D float Tensor with values ranging from [0, 1).
      """
      with tf.name_scope(values=[image_buffer], name=scope,
                         default_name='decode_png'):
        # Decode the string as an PNG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_png(image_buffer, channels=num_channels)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image



    def image_preprocessing(self, image_buffer, img_size, num_channels):
      """Decode and preprocess one image for evaluation or training.
    
      Args:
        image_buffer: JPEG encoded string Tensor
    
      Returns:
        3-D float Tensor containing an appropriately scaled image
      """

      image = self.decode_png(image_buffer, num_channels)

      # Image size is unknown until run-time so we need to resize the image to specific size
      image = tf.image.resize_images(image, size=img_size)

      # Finally, rescale to [-1,1] instead of [0, 1)
      #image = tf.subtract(image, 0.5)
      #image = tf.multiply(image, 2.0)
      return image


    def parse_example_proto(self, example_serialized):
      """Parses an Example proto containing a training example of an image.
    
      The output of the build_image_data.py image preprocessing script is a input_data
      containing serialized Example protocol buffers. Each Example proto contains
      the following fields:
    
        image/height: 750
        image/width: 250
        image/channels: 1
        image/class/encoded: <PNG encoded string>
        image/class/filename: 'knee pad'
        image/class/channels: 1
        image/format: 'PNG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <PNG encoded string>
    
      Args:
        example_serialized: scalar Tensor tf.string containing a serialized
          Example protocol buffer.
    
      Returns:
        image_buffer: Tensor tf.string containing the contents of a PNG file.
        label_buffer: Tensor tf.string containing the contents of a PNG mask/groundtruth file.
      """
      # Dense features in Example proto.
      feature_map = {
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'image/class/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
          'image/name': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      }

      features = tf.parse_single_example(example_serialized, feature_map)

      return features['image/encoded'], features['image/class/encoded'], features['image/name']


