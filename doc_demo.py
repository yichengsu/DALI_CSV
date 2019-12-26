from __future__ import division

import collections
import os
import types
from random import shuffle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import \
    DALIClassificationIterator as PyTorchIterator


class ExternalInputIterator(object):
    def __init__(self, batch_size, device_id, num_gpus):
        self.images_dir = "./images/"
        self.batch_size = batch_size
        with open(self.images_dir + "file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line is not '']
        # whole data set size
        self.data_set_len = len(self.files)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        # shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.i].split(' ')
            f = open(self.images_dir + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    @property
    def size(self,):
        return self.data_set_len

    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=240, resize_y=240)
        self.cast = ops.Cast(device="gpu",
                             dtype=types.UINT8)
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cast(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


def show_images(image_batch):
    columns = 3
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(len(image_batch)):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch[j, :, :, :].cpu())


batch_size = 3
epochs = 3

eii = ExternalInputIterator(batch_size, 0, 1)
pipe = ExternalSourcePipeline(batch_size=batch_size, num_threads=2, device_id=0,
                              external_data=eii)
pii = PyTorchIterator(pipe, size=eii.size, last_batch_padded=True, fill_last_batch=False)

if not os.path.exists('res'):
    os.mkdir('res')
for e in range(epochs):
    for i, data in enumerate(pii):
        print("epoch: {}, iter {}, real batch size: {}".format(e, i, len(data[0]["data"])))
        show_images(data[0]["data"])
        plt.savefig('res/demo_epoch{}_iter{}.jpg'.format(e, i))
    pii.reset()
