import torch
import random


class ForeverDataIterator:
    """A data iterator that will never stop producing data"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class ConnectedDataIterator:
    def __init__(self, dataloader_list, batch_size):
        self.dataloader_list = dataloader_list
        self.batch_size = batch_size

        self.length = len(self.dataloader_list)
        self.iter_list = [iter(loader) for loader in self.dataloader_list]
        self.available_set = set([i for i in range(self.length)])

    def append(self, index_list):
        self.available_set = self.available_set | set(index_list)

    def keep(self, index_list):
        self.available_set = set(index_list)

    def remove(self, index_list):
        self.available_set = self.available_set - set(index_list)

    def reset(self):
        self.available_set = set([i for i in range(len(self.dataloader_list))])

    def __next__(self):
        data_sum = []
        label_sum = []
        for i in self.available_set:
            tries, max_tries = 0, 32
            while True:
                try:
                    data, label, *_ = next(self.iter_list[i])
                    break
                except StopIteration:
                    self.iter_list[i] = iter(self.dataloader_list[i])
                    tries += 1
                    if tries >= max_tries:
                        data = label = None
                        break
            if data is None:
                continue
            data_sum.append(data)
            label_sum.append(label)

        if not data_sum:
            raise RuntimeError("All loaders yielded no data; check dataset sizes or drop_last settings")

        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)
        #rand_index = random.sample([i for i in range(len(data_sum))], self.batch_size)
        actual_batch_size = min(self.batch_size, len(data_sum))
        rand_index = random.sample([i for i in range(len(data_sum))], actual_batch_size)

        return data_sum[rand_index], label_sum[rand_index]

    def next(self, all=False, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        data_sum = []
        label_sum = []
        for i in self.available_set:
            tries, max_tries = 0, 32
            while True:
                try:
                    data, label, *_ = next(self.iter_list[i])
                    break
                except StopIteration:
                    self.iter_list[i] = iter(self.dataloader_list[i])
                    tries += 1
                    if tries >= max_tries:
                        data = label = None
                        break
            if data is None:
                continue

            data_sum.append(data)
            label_sum.append(label)

        if not data_sum:
            raise RuntimeError("All loaders yielded no data; check dataset sizes or drop_last settings")

        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)

        if all:
            return data_sum, label_sum

        #rand_index = random.sample([i for i in range(len(data_sum))], batch_size)
        actual_batch_size = min(batch_size, len(data_sum))
        rand_index = random.sample([i for i in range(len(data_sum))], actual_batch_size)

        return data_sum[rand_index], label_sum[rand_index]