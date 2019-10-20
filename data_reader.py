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

import collections
import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['query_camera', 'target'])
Input = collections.namedtuple('Input', ['context_frames', 'context_cameras', 'query_camera', 'target'])
TaskData = collections.namedtuple('TaskData', ['inputs', 'target'])


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


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)

    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def _get_randomized_indices(context_size, sequence_size):
    """Generates randomized indices into a sequence of a specific length."""
    if context_size is None:
        maximum_context_size = min(sequence_size-1, 20)
        context_size = tf.random.uniform([1], 1, maximum_context_size, dtype=tf.int32)
    else:
        context_size = tf.constant(context_size, shape=[1], dtype=tf.int32)
    example_size = context_size + 1

    indices = tf.range(0, sequence_size)
    indices = tf.random.shuffle(indices)
    indices = tf.slice(indices, begin=[0], size=example_size)

    return indices, example_size


def data_reader(dataset,
                root,
                mode,
                batch_size,
                context_size=None,
                custom_frame_size=None,
                shuffle_buffer_size=256,
                num_parallel_reads=4,
                seed=None):

    if dataset not in _DATASETS:
        raise ValueError('Unrecognized dataset {} requested. Available datasets '
                         'are {}'.format(dataset, _DATASETS.keys()))

    if mode not in _MODES:
        raise ValueError('Unsupported mode {} requested. Supported modes '
                         'are {}'.format(mode, _MODES))

    dataset_info = _DATASETS[dataset]

    if context_size is not None and context_size >= dataset_info.sequence_size:
        raise ValueError(
            'Maximum support context size for dataset {} is {}, but '
            'was {}.'.format(
                dataset, dataset_info.sequence_size-1, context_size))

    tf.random.set_seed(seed)

    basepath = dataset_info.basepath
    file_pattern = os.path.join(root, basepath, mode, '*.tfrecord')
    files = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    raw_dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=num_parallel_reads,
        num_parallel_calls=AUTOTUNE).repeat().shuffle(shuffle_buffer_size, seed=seed)

    feature_map = {
    'frames': tf.io.FixedLenFeature(
        shape=dataset_info.sequence_size, dtype=tf.string),
    'cameras': tf.io.FixedLenFeature(
        shape=[dataset_info.sequence_size * _NUM_RAW_CAMERA_PARAMS],
        dtype=tf.float32)
    }

    def _parse_function(example):
        return tf.io.parse_single_example(example, feature_map)

    parsed_dataset = raw_dataset.map(_parse_function,
                                     num_parallel_calls=AUTOTUNE).batch(batch_size)

    def _preprocess_fn(example):
        frames = example['frames']
        raw_pose_params = example['cameras']

        indices, example_size = _get_randomized_indices(context_size, dataset_info.sequence_size)

        frames = tf.gather(frames, indices, axis=1)
        frames = tf.map_fn(
            _convert_frame_data, tf.reshape(frames, [-1]),
            dtype=tf.float32, back_prop=False)
        dataset_image_dimensions = tuple(
            [dataset_info.frame_size] * 2 + [_NUM_CHANNELS])
        frames = tf.reshape(
            frames, (-1, example_size[0]) + dataset_image_dimensions)
        if (custom_frame_size and
            custom_frame_size != dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (custom_frame_size,) * 2 + (_NUM_CHANNELS,)
            frames = tf.image.resize(
                frames, new_frame_dimensions[:2])
            frames = tf.reshape(
                frames, (-1, example_size[0]) + new_frame_dimensions)

        raw_pose_params = tf.reshape(
            raw_pose_params,
            [-1, dataset_info.sequence_size, _NUM_RAW_CAMERA_PARAMS])
        raw_pose_params = tf.gather(raw_pose_params, indices, axis=1)
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)

        context_frames = frames[:, :-1]
        context_cameras = cameras[:, :-1]
        target = frames[:, -1]
        query_camera = cameras[:, -1]
        # context = Context(frames=context_frames, cameras=context_cameras)
        query = Query(query_camera=query_camera, target=target)
        inputs = Input(context_frames=context_frames,
                       context_cameras=context_cameras,
                       query_camera=query_camera,
                       target=target)


        return TaskData(inputs=inputs, target=target)

    preprocessed_dataset = parsed_dataset.map(_preprocess_fn,
                                              num_parallel_calls=AUTOTUNE)

    return preprocessed_dataset.prefetch(buffer_size=AUTOTUNE)
