## Datasets used to train Generative Query Networks (GQNs) in the ‘Neural Scene Representation and Rendering’ paper.


The following version of the datasets are available:

*   __rooms_ring_camera__. Scenes of a variable number of random objects
    captured in a square room of size 7x7 units. Wall textures, floor textures
    as well as the shapes of the objects are randomly chosen within a fixed pool
    of discrete options. There are 5 possible wall textures (red, green, cerise,
    orange, yellow), 3 possible floor textures (yellow, white, blue) and 7
    possible object shapes (box, sphere, cylinder, capsule, cone, icosahedron
    and triangle). Each scene contains 1, 2 or 3 objects. In this simplified
    version of the dataset, the camera only moves on a fixed ring and always
    faces the center of the room. This is the ‘easiest’ version of the dataset,
    use version for fast training.

*   __rooms_free_camera_no_object_rotations__. As in __rooom_ring_camera__,
    except the camera moves freely. However the objects themselves do not rotate
    around their axes, which makes the modeling task somewhat easier. This
    version is ‘medium’ difficulty.

*   __rooms_free_camera_with_object_rotations__. As in
    __rooms_free_camera_no_object_rotations__, the camera moves freely, however
    objects can rotate around their vertical axes across scenes. This is the
    ‘hardest’ version of the dataset.

*   __jaco__. a reproduction of the robotic Jaco arm is placed in the middle of
    the room along with one spherical target object. The arm has nine joints. As
    above, the appearance of the room is modified for each episode by randomly
    choosing a different texture for the walls and floor from a fixed pool of
    options. In addition, we modify both colour and position of the target
    randomly. Finally, the joint angles of the arm are also initialised at
    random within a range of physically sensible positions.

*   __shepard_metzler_5_parts__. Each object is composed of 7 randomly coloured
    cubes that are positioned by a self-avoiding random walk in 3D grid. As
    above, the camera is parametrised by its position, yaw and pitch, however it
    is constrained to only move around the object at a fixed distance from its
    centre. This is the ‘easy’ version of the dataset, where each object is
    composed of only 5 parts.

*   __shepard_metzler_7_parts__. This is the ‘hard’ version of the above
    dataset, where each object is composed of 7 parts.

*   __mazes__. Random mazes that were created using an OpenGL-based [DeepMind
    Lab](https://github.com/deepmind/lab) game engine (Beattie et al., 2016).
    Each maze is constructed out of an underlying 7 by 7 grid, with walls
    falling on the boundaries of the grid locations. However, the agent can be
    positioned at any continuous position in the maze. The mazes contain 1 or 2
    rooms, with multiple connecting corridors. The walls and floor textures of
    each maze are determined by random uniform sampling from a predefined set of
    textures.

### Usage example

To select what dataset to load, instantiate a reader passing the correct
`version` argument. Note that the constructor will set up all the queues used by
the reader. To get tensors call `read` on the data reader passing in the desired
batch size.

```python
  import tensorflow as tf

  root_path = 'path/to/datasets/root/folder'
  data_reader = DataReader(dataset='jaco', context_size=5, root=root_path)
  data = data_reader.read(batch_size=12)

  with tf.train.SingularMonitoredSession() as sess:
    d = sess.run(data)
```

### Download

Raw data files referred to in this document are available to download
[here](https://console.cloud.google.com/storage/gqn-dataset). To download the
datasets you can use
the [`gsutil cp`](https://cloud.google.com/storage/docs/gsutil/commands/cp)
command; see also the `gsutil` [installation instructions]
(https://cloud.google.com/storage/docs/gsutil_install).


### Notes

This is not an official Google product.
