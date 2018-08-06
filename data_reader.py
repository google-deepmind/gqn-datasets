# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal data reader for GQN TFRecord datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import tensorflow as tf
nest = tf.contrib.framework.nest

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])


_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)
_NUM_CHANNELS = 3
_NUM_RAW_CAMERA_PARAMS = 5
_MODES = ('train', 'test')


def _get_dataset_files(dateset_info, mode, root):
  """Generates lists of files for a given dataset version."""
  basepath = dateset_info.basepath
  base = os.path.join(root, basepath, mode)
  if mode == 'train':
    num_files = dateset_info.train_size
  else:
    num_files = dateset_info.test_size

  length = len(str(num_files))
  template = '{:0%d}-of-{:0%d}.tfrecord' % (length, length)
  return [os.path.join(base, template.format(i + 1, num_files))
          for i in range(num_files)]


def _convert_frame_data(jpeg_data):
  decoded_frames = tf.image.decode_jpeg(jpeg_data)
  return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


class DataReader(object):
  """Minimal queue based TFRecord reader.

  You can use this reader to load the datasets used to train Generative Query
  Networks (GQNs) in the 'Neural Scene Representation and Rendering' paper.
  See README.md for a description of the datasets and an example of how to use
  the reader.
  """

  def __init__(self,
               dataset,
               context_size,
               root,
               mode='train',
               # Optionally reshape frames
               custom_frame_size=None,
               # Queue params
               num_threads=4,
               capacity=256,
               min_after_dequeue=128,
               seed=None):
    """Instantiates a DataReader object and sets up queues for data reading.

    Args:
      dataset: string, one of ['jaco', 'mazes', 'rooms_ring_camera',
          'rooms_free_camera_no_object_rotations',
          'rooms_free_camera_with_object_rotations', 'shepard_metzler_5_parts',
          'shepard_metzler_7_parts'].
      context_size: integer, number of views to be used to assemble the context.
      root: string, path to the root folder of the data.
      mode: (optional) string, one of ['train', 'test'].
      custom_frame_size: (optional) integer, required size of the returned
          frames, defaults to None.
      num_threads: (optional) integer, number of threads used to feed the reader
          queues, defaults to 4.
      capacity: (optional) integer, capacity of the underlying
          RandomShuffleQueue, defualts to 256.
      min_after_dequeue: (optional) integer, min_after_dequeue of the underlying
          RandomShuffleQueue, defualts to 128.
      seed: (optional) integer, seed for the random number generators used in
          the reader.

    Raises:
      ValueError: if the required version does not exist; if the required mode
         is not supported; if the requested context_size is bigger than the
         maximum supported for the given dataset version.
    """

    if dataset not in _DATASETS:
      raise ValueError('Unrecognized dataset {} requested. Available datasets '
                       'are {}'.format(dataset, _DATASETS.keys()))

    if mode not in _MODES:
      raise ValueError('Unsupported mode {} requested. Supported modes '
                       'are {}'.format(mode, _MODES))

    self._dataset_info = _DATASETS[dataset]

    if context_size >= self._dataset_info.sequence_size:
      raise ValueError(
          'Maximum support context size for dataset {} is {}, but '
          'was {}.'.format(
              dataset, self._dataset_info.sequence_size-1, context_size))

    self._context_size = context_size
    # Number of views in the context + target view
    self._example_size = context_size + 1
    self._custom_frame_size = custom_frame_size

    with tf.device('/cpu'):
      file_names = _get_dataset_files(self._dataset_info, mode, root)
      filename_queue = tf.train.string_input_producer(file_names, seed=seed)
      reader = tf.TFRecordReader()

      read_ops = [self._make_read_op(reader, filename_queue)
                  for _ in range(num_threads)]

      dtypes = nest.map_structure(lambda x: x.dtype, read_ops[0])
      shapes = nest.map_structure(lambda x: x.shape[1:], read_ops[0])

      self._queue = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=dtypes,
          shapes=shapes,
          seed=seed)

      enqueue_ops = [self._queue.enqueue_many(op) for op in read_ops]
      tf.train.add_queue_runner(tf.train.QueueRunner(self._queue, enqueue_ops))

  def read(self, batch_size):
    """Reads batch_size (query, target) pairs."""
    frames, cameras = self._queue.dequeue_many(batch_size)
    context_frames = frames[:, :-1]
    context_cameras = cameras[:, :-1]
    target = frames[:, -1]
    query_camera = cameras[:, -1]
    context = Context(cameras=context_cameras, frames=context_frames)
    query = Query(context=context, query_camera=query_camera)
    return TaskData(query=query, target=target)

  def _make_read_op(self, reader, filename_queue):
    """Instantiates the ops used to read and parse the data into tensors."""
    _, raw_data = reader.read_up_to(filename_queue, num_records=16)
    feature_map = {
        'frames': tf.FixedLenFeature(
            shape=self._dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.FixedLenFeature(
            shape=[self._dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
            dtype=tf.float32)
    }
    example = tf.parse_example(raw_data, feature_map)
    indices = self._get_randomized_indices()
    frames = self._preprocess_frames(example, indices)
    cameras = self._preprocess_cameras(example, indices)
    return frames, cameras

  def _get_randomized_indices(self):
    """Generates randomized indices into a sequence of a specific length."""
    indices = tf.range(0, self._dataset_info.sequence_size)
    indices = tf.random_shuffle(indices)
    indices = tf.slice(indices, begin=[0], size=[self._example_size])
    return indices

  def _preprocess_frames(self, example, indices):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    frames = tf.gather(frames, indices, axis=1)
    frames = tf.map_fn(
        _convert_frame_data, tf.reshape(frames, [-1]),
        dtype=tf.float32, back_prop=False)
    dataset_image_dimensions = tuple(
        [self._dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
    frames = tf.reshape(
        frames, (-1, self._example_size) + dataset_image_dimensions)
    if (self._custom_frame_size and
        self._custom_frame_size != self._dataset_info.frame_size):
      frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
      new_frame_dimensions = (self._custom_frame_size,) * 2 + (_NUM_CHANNELS,)
      frames = tf.image.resize_bilinear(
          frames, new_frame_dimensions[:2], align_corners=True)
      frames = tf.reshape(
          frames, (-1, self._example_size) + new_frame_dimensions)
    return frames

  def _preprocess_cameras(self, example, indices):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, self._dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
    raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
    pos = raw_pose_params[:, :, 0:3]
    yaw = raw_pose_params[:, :, 3:4]
    pitch = raw_pose_params[:, :, 4:5]
    cameras = tf.concat(
        [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
    return cameras
