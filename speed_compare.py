import os
import time
import types
from random import shuffle

import numpy as np
import pandas as pd
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator


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
        labels = []

        for idx in order:
            filename = self.csv['image_name'][idx]
            label = self.csv['MOS'][idx]

            with open(os.path.join(self.images_folder, filename), 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))

            if len(batch) == self.batch_size:
                yield (batch, labels)
                batch = []
                labels = []

        if len(batch) > 0:
            yield (batch, labels)

    @property
    def size(self):
        return self.data_set_len


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()

        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.norm = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()

        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.norm(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = next(self.iterator)
            if len(images) < self.batch_size:
                # just add last one
                tmp_images = images[-1]
                tmp_label = labels[-1]
                for _ in range(self.batch_size-len(images)):
                    images.append(tmp_images)
                    labels.append(tmp_label)
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)

        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, image_dir):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id=0)
        self.input = ops.FileReader(file_root=image_dir, random_shuffle=True, initial_fill=256)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.resize(images)
        images = self.cmnp(images)
        return (images, labels)


if __name__ == "__main__":

    BATCH_SIZE = 128
    IMAGE_FOLDER = './data'
    NUM_WORKERS = 16

    # pytorch dataloader
    transformer = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(IMAGE_FOLDER, transform=transformer)
    dataloader = Data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    start = time.time()
    length = 0
    for x, y in dataloader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        length += len(x)
    end = time.time()
    print('Pytorch dataloader: {} items in {:.2f}s({:.2f}items/s)'.format(length, end-start, length/(end-start)))

    # ops.FileReader + HybridPipeline
    pipe = SimplePipeline(BATCH_SIZE, NUM_WORKERS, IMAGE_FOLDER)
    pipe.build()
    daliloder = DALIClassificationIterator(pipe, size=pipe.epoch_size("Reader"),
                                           last_batch_padded=True, fill_last_batch=False)
    start = time.time()
    length = 0
    for data in daliloder:
        x = data[0]['data'].cuda(non_blocking=True)
        y = data[0]['label'].squeeze().long().cuda(non_blocking=True)
        length += len(x)
    end = time.time()
    print('DALI FileReader: {} items in {:.2f}s({:.2f}items/s)'.format(length, end-start, length/(end-start)))

    # csv
    IMAGE_FOLDER = os.path.join(IMAGE_FOLDER, 'koniq10k')
    MOS_FILE = './mos.csv'
    csvii = CSVInputIterator(BATCH_SIZE, IMAGE_FOLDER, MOS_FILE, shuffle=True)
    pipe = ExternalSourcePipeline(batch_size=BATCH_SIZE, num_threads=NUM_WORKERS, device_id=0,
                                  external_data=csvii)
    dalicsvloader = DALIGenericIterator(pipe, output_map=['data', 'label'], size=csvii.size,
                                        last_batch_padded=True, fill_last_batch=False)
    start = time.time()
    length = 0
    for data in dalicsvloader:
        x = data[0]['data'].cuda(non_blocking=True)
        y = data[0]['label'].squeeze().long().cuda(non_blocking=True)
        length += len(x)
    end = time.time()
    print('DALI CSV loader: {} items in {:.2f}s({:.2f}items/s)'.format(length, end-start, length/(end-start)))
