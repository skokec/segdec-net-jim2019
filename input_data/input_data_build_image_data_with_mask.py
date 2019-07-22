# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts image data to TFRecords file format with Example protos.

The image data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/encoded: string containing PNG encoded class mask 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure, iterate_structure

from cStringIO import StringIO
from PIL import Image

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, img_channels, mask_filename, mask_buffer, mask_channels, image_format, height, width, naming_fn = None):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, PNG/JPEG encoding of RGB image
    mask_filename: string, path to groundtruth mask file
    mask_buffer: string, PNG/JPEG encoding of mask image
    image_format: string, format of encoded images, e.g., PNG or JPEG
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """

  feat = {
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'image/channels': _int64_feature(img_channels),
    'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
    'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
    'image/name': _bytes_feature(
      tf.compat.as_bytes(naming_fn(filename) if naming_fn is not None else  os.path.abspath(filename))),
    'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}

  if mask_buffer is not None:
    feat['image/class/encoded'] = _bytes_feature(tf.compat.as_bytes(mask_buffer))
    feat['image/class/channels'] = _int64_feature(mask_channels)

  if mask_filename is not None:
    feat['image/class/filename'] = _bytes_feature(tf.compat.as_bytes(os.path.basename(mask_filename)))


  example = tf.train.Example(features=tf.train.Features(feature=feat))
  return example


def _process_image(filename, out_format, resize = None, dilate = None, require_binary_output = False):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    out_format: string, output format type e.g., 'PNG', 'JPEG' 
  Returns:
    image_buffer: string, encoding of image in out_format
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    raw_image_data = f.read()

  # Convert any format to PNG for consistency.
  pil_img = Image.open(StringIO(raw_image_data))

  # dilate image if requested so - create structering element of appropriate size
  if dilate is not None:
    dilation_se = iterate_structure(generate_binary_structure(2,1),(int)((dilate-1)/2))
    im = binary_dilation(np.array(pil_img), structure=dilation_se)
    pil_img = Image.fromarray(np.uint8(im)*255)

  if resize is not None:
    pil_img = pil_img.resize(resize[::-1]) # NOTE: use reversed order of resize to make input consistent with tensorflow

  # if output should be in binary then we must do binarization to remove interpolation values from resize
  if require_binary_output:
    im = (np.array(pil_img) > 0)
    pil_img = Image.fromarray(np.uint8(im) * 255)

  image_data = StringIO()
  pil_img.save(image_data, out_format)

  height = pil_img.size[1]
  width = pil_img.size[0]
  if pil_img.mode in ['RGBA', 'CMYK']:
    num_chanels = 4
  elif pil_img.mode in ['RGB','LAB','HSV','YCbCr']:
    num_chanels = 3
  else:
    num_chanels = 1

  return image_data.getvalue(), height, width, num_chanels


def _process_image_files_batch(image_format, thread_index, ranges, name, filenames, masks, num_shards, output_directory, resize = None, naming_fn = None, dilate = None, require_binary_output = False, export_images = False):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    image_format: string, output format type e.g., 'PNG', 'JPEG'
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    masks: list of strings; each string is a path to an groundtruth mask file
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    export_folder_imgs = None
    export_folder_masks = None
    if export_images:
        export_folder = os.path.join(output_directory, 'export')
        export_folder_imgs = os.path.join(export_folder, 'imgs')
        export_folder_masks = os.path.join(export_folder, 'masks')

        try:
            os.makedirs(export_folder_imgs)
        except:
            pass

        try:
            os.makedirs(export_folder_masks)
        except:
            pass


    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      mask_filename = masks[i]

      try:
        image_buffer, img_height, img_width, img_channels = _process_image(filename, image_format, resize)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % filename)
        continue

      try:
        mask_buffer, mask_height, mask_width, mask_channels = _process_image(mask_filename, image_format, resize, dilate,require_binary_output=require_binary_output)

      except Exception as e:
        print('WARNING: No mask found for %s - using empty mask instead' % filename)

        # Generate dummy mask
        pil_img = Image.fromarray(np.zeros([img_height, img_width], dtype=np.uint8))
        image_data = StringIO()
        pil_img.save(image_data, image_format)

        mask_buffer = image_data.getvalue()
        mask_height = pil_img.size[1]
        mask_width = pil_img.size[0]
        mask_channels = 1

      assert img_height == mask_height
      assert img_width == mask_width

      example = _convert_to_example(filename, image_buffer, img_channels, mask_filename, mask_buffer, mask_channels, image_format, img_height, img_width, naming_fn=naming_fn)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      # if export_images is set then we need to export image and mask as raw image into appropriate folder
      if export_images and export_folder_imgs is not None and export_folder_masks is not None:

          # get name of sample using img name and its last folder parent
          part_name = os.path.basename(os.path.dirname(filename))
          part_id = os.path.splitext(os.path.basename(filename))[0]

          export_img_name = str.format("{0}_{1}.{2}",part_name, part_id, image_format.lower() )
          export_mask_name = str.format("{0}_{1}_label.{2}", part_name, part_id, image_format.lower())

          # export both img and its mask
          Image.open(StringIO(image_buffer)).save(os.path.join(export_folder_imgs, export_img_name))
          Image.open(StringIO(mask_buffer)).save(os.path.join(export_folder_masks, export_mask_name))


      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, masks, output_directory, num_shards, num_threads, resize = None, naming_fn = None, dilate = None, require_binary_output = False, export_images = False):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    masks: list of strings; each string is a path to an groundtruth mask file
    output_directory: string, output folder path 
    num_shards: integer number of shards for this data set.
    num_threads: integer number of threads to use, NOTE: must be num_shards % num_threads == 0
  """
  if masks is not None:
    assert len(filenames) == len(masks)

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  image_format = 'PNG'

  threads = []
  for thread_index in range(len(ranges)):
    args = (image_format, thread_index, ranges, name, filenames, masks, num_shards, output_directory, resize, naming_fn, dilate, require_binary_output, export_images)
    _process_image_files_batch(*args)
    #t = threading.Thread(target=_process_image_files_batch, args=args)
    #t.start()
    #threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir_list, data_ext, mask_pattern, ignore_non_positive_masks = False):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir_list: array of string, list of paths to the root directory of images.

      Assumes that the image data set resides in JPEG files located in
      the following directory structure.

        data_dir[0]/another-image.JPEG
        data_dir[1]/my-image.jpg

      with corresponding mask files in the same folders
        
        data_dir[0]/another-image_mask.png
        data_dir[1]/my-image_mask.png
        
    data_ext: string, extension of images
    mask_pattern: tuple of string, with mask_pattern[0] string replace pattern 
      and mask_pattern[1] replace string, e.g., mask_pattern = ('.jpg', '_mask.png')
    ignore_non_positive_masks: boolean, ignores files that have only zero mask values

  Returns:
    filenames: list of strings; each string is a path to an image file.
    mask_filenames: list of strings; each string is a path to an image mask file.
  """

  filenames = []
  mask_filenames = []

  # Construct the list of JPEG files and labels.
  for data_dir in data_dir_list:
    print('Determining list of input files and labels from %s.' % data_dir)

    jpeg_file_path = '%s/*%s' % (data_dir, data_ext)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    # remove files that are actual labels !!
    matching_files = [f for f in matching_files if not f.endswith(mask_pattern[1])]

    filenames.extend(matching_files)

    # Find corresponding mask files
    matching_files_masks = [filename.replace(mask_pattern[0],mask_pattern[1]) for filename in matching_files]

    mask_filenames.extend(matching_files_masks)

  if ignore_non_positive_masks:
    select = [np.any(Image.open(m)) for m in mask_filenames if os.path.exists(m)]
    filenames = [f for f,s in zip(filenames,select) if s]
    mask_filenames = [m for m, s in zip(mask_filenames, select) if s]

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  mask_filenames = [mask_filenames[i] for i in shuffled_index]

  print('Found %d image files across all folders.' % (len(filenames)))

  return filenames, mask_filenames


def _process_dataset(name, directory_list, data_extension, mask_patterns, output_directory, num_shards, num_threads, resize = None, ignore_non_positive_masks = False, naming_fn = None, dilate = None, require_binary_output = False, export_images = False):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames, masks = _find_image_files(directory_list, data_extension, mask_patterns, ignore_non_positive_masks)
  _process_image_files(name, filenames, masks, output_directory, num_shards, num_threads, resize=resize, naming_fn = naming_fn, dilate = dilate, require_binary_output=require_binary_output, export_images = export_images)

