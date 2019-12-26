import argparse
import os
import time
import types
from random import shuffle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import pandas as pd
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class CSVInputIterator(object):
    def __init__(self, batch_size, images_folder, csv_path, shuffle=True, device_id=0, num_gpus=1):
        self.images_folder = images_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csv = pd.read_csv(csv_path)
        # whole data set size
        self.data_set_len = len(self.csv)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.csv = self.csv.iloc[self.data_set_len * device_id // num_gpus:
                                 self.data_set_len * (device_id + 1) // num_gpus]

    def __iter__(self):
        order = list(range(len(self.csv)))
        if self.shuffle:
            shuffle(order)

        batch = []
        labels1 = []
        labels2 = []

        for idx in order:
            filename = self.csv['image'][idx]
            label1 = self.csv['label1'][idx]
            label2 = self.csv['label2'][idx]

            with open(os.path.join(self.images_folder, filename), 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels1.append(np.array([label1], dtype=np.uint8))
            labels2.append(np.array([label2], dtype=np.uint8))

            if len(batch) == self.batch_size:
                yield (batch, labels1, labels2)
                batch = []
                labels1 = []
                labels2 = []

        if len(batch) > 0:
            yield (batch, labels1, labels2)

    @property
    def size(self):
        return self.data_set_len


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.ExternalSource()
        self.input_label1 = ops.ExternalSource()
        self.input_label2 = ops.ExternalSource()

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cast = ops.Cast(device="gpu", dtype=types.UINT8)
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels1 = self.input_label1()
        self.labels2 = self.input_label2()

        images = self.decode(self.jpegs)
        images = self.res(images)
        # images = self.cast(images)
        return (images, self.labels1, self.labels2)

    def iter_setup(self):
        try:
            (images, labels1, labels2) = next(self.iterator)
            if len(images) < self.batch_size:
                # just add last one
                tmp_images = images[-1]
                tmp_label1 = labels1[-1]
                tmp_label2 = labels2[-1]
                for _ in range(self.batch_size-len(images)):
                    images.append(tmp_images)
                    labels1.append(tmp_label1)
                    labels2.append(tmp_label2)
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels1, labels1)
            self.feed_input(self.labels2, labels2)

        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


def show_images(data, batch_size):
    columns = batch_size
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(15, (15 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(len(data['data'])):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(data['data'][j, :, :, :].cpu())
        plt.title('Label_{}_{}'.format(data['label1'][j].item(), data['label2'][j].item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='python dali_cav.py',
                                     description='DALI PyTorch CSV')
    parser.add_argument('-images_folder', default='images', type=str,
                        help='images folder (default: images)')
    parser.add_argument('-mos_file', default='images_info.csv', type=str,
                        help='mos file (default: images_info.csv)')
    parser.add_argument('-batch_size', default=3, type=int,
                        help='batch size (default: 3)')
    parser.add_argument('-epochs', default=3, type=int,
                        help='epochs (default: 3)')
    args = parser.parse_args()

    csvii = CSVInputIterator(args.batch_size, args.images_folder, args.mos_file, shuffle=False)
    pipe = ExternalSourcePipeline(batch_size=args.batch_size, num_threads=2, device_id=0,
                                  external_data=csvii)
    pii = DALIGenericIterator(pipe, output_map=['data', 'label1', 'label2'], size=csvii.size,
                              last_batch_padded=True, fill_last_batch=False)

    if not os.path.exists('res'):
        os.mkdir('res')
    for e in range(args.epochs):
        for i, data in enumerate(pii):
            print("epoch: {}, iter {}, real batch size: {}".format(e, i, len(data[0]["data"])))
            show_images(data[0], args.batch_size,)
            plt.savefig('res/csv_epoch{}_iter{}.jpg'.format(e, i))
        pii.reset()
