# PyTorch DataLoader with DALI and CSV
This repo shows a demo of how to use DALI(v0.16.0) to read images and label from the CSV config file.
`./images` folder provide five images as a small dataset.

*Allow me to complain first.*

`./doc_demo.py` comes from the [document](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-external_input.html) of DALI. 

You can run this demo like this and it will show the results in `./res/`:
```
python doc_demo.py
```

To be honest, `DALIGenericIterator` does **not** implement the features described in the documentation. Especially these two parameters of it: `fill_last_batch` and `last_batch_padded`.

With the data set [1,2,3,4,5,6,7] and the batch size 2:

| `fill_last_batch` | `last_batch_padded` | last batch | next iteration | realize |
| :---------------: | :-----------------: | :--------: | :------------: | :-----: |
|       False       |        True         |    [7]     |     [1, 2]     |  **×**  |
|       False       |        False        |    [7]     |     [2, 3]     |    √    |
|       True        |        True         |   [7, 7]   |     [1, 2]     |  **×**  |
|       True        |        False        |   [7, 1]   |     [2, 3]     |    √    |

I also looked at the source code in [github](https://github.com/NVIDIA/DALI/blob/e354de1bebd34bcd365ce38ac900e0295d5625dc/dali/python/nvidia/dali/plugin/pytorch.py), and these two parameters did not achieve the claimed function.

`ExternalInputIterator` also makes me confuse.
``` python
def __next__(self):
        ...
        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            ...
            self.i = (self.i + 1) % self.n
        ...
```
It never `raise StopIteration` because `self.i = (self.i + 1) % self.n`. This also makes it **impossible** to cooperate with the above functions `DALIGenericIterator`. The next epoch will never start at the beginning.
It doesn't seem to be a problem when used on the training set, but it feels weird when used on the test set, because you don't know where it started, although it may not affect the final result.

Maybe it has to make some compromises for better compatibility with Python 2.x. But I hope that DALI can provide better design on this issue in the future.

*Next is how to use DALI to read images and label from the CSV config file.*

I use a different philosophy from [PyTorch](https://pytorch.org/docs/1.1.0/_modules/torch/utils/data/sampler.html). I wrote about it in my [blog](https://yichengsu.github.io/2019/12/Iterator-and-Iterable/).

You can run `dali_csv.py` like this and it will also show the results in `./res/`:
```
python dali_csv.py
```
or provide some parameters:
```
CUDA_VISIBLE_DEVICES=3 python dali_csv.py -batch_size 2 -epochs 2
```

It has the following advantages:
- Using a csv file, you can easily separate the training set from the test set
- Provides the function of shuffle
- Can return multi-labels
- Read the complete dataset for each epoch

I highly recommend `DALIGenericIterator(..., last_batch_padded=True/False, fill_last_batch=False)`. It will always read the complete dataset for each epoch. `ffill_last_batch=True` will make the last epoch have a lot of duplicate data or bring some other mistakes.

Because I made a few changes to the original structure, it most likely does **not support** Python 2.x. You can also easily merge the two files and use the original structure.

With Intel(R) Xeon(R) CPU E5-2650 v4, 1 TITAN Xp GPU, I compared the speed of these three situations using the [KonIQ-10K](http://database.mmsp-kn.de/koniq-10k-database.html) dataset which has 10,073 images.

|                     | 4 threads            | 8 threads            | 16 threads           |
| ------------------- | -------------------- | -------------------- | -------------------- |
| PyTorch dataloader  | 165.55s(62.66imgs/s) | 96.07s(107.97imgs/s) | 53.75s(192.99imgs/s) |
| DALI ops.FileReader | 45.92s(225.89imgs/s) | 24.76s(418.98imgs/s) | 15.39s(673.96imgs/s) |
| DALI CSV loader     | 44.71s(225.30imgs/s) | 24.77s(406.62imgs/s) | 14.82s(679.72imgs/s) |

Although the server I used is always busy and all data is stored on disk, it still shows very promising speed.

This repo was inspired by [tanglang96](https://github.com/tanglang96/DataLoaders_DALI).
