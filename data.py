import os
import time
import random
import skimage.io
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision as tv
from torchvision.datasets import CIFAR100

class self_Dataset(Dataset):
    def __init__(self, data, label=None):
        super(self_Dataset, self).__init__()

        self.data = data
        self.label = label
    def __getitem__(self, index):
        data = self.data[index]
        # data = np.moveaxis(data, 3, 1)
        # data = data.astype(np.float32)

        if self.label is not None:
            label = self.label[index]
            # print(label)
            # label = torch.from_numpy(label)
            # label = torch.LongTensor([label])
            return data, label
        else:
            return data, 1
    def __len__(self):
        return len(self.data)

def count_data(data_dict):
    num = 0
    for key in data_dict.keys():
        num += len(data_dict[key])
    return num

class self_DataLoader(Dataset):
    def __init__(self, root, train=True, dataset='cifar100', seed=1, nway=5):
        super(self_DataLoader, self).__init__()

        self.seed = seed
        self.nway = nway
        self.num_labels = 100
        self.input_channels = 3
        self.size = 32

        self.transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.5071, 0.4866, 0.4409], 
                [0.2673, 0.2564, 0.2762])
            ])

        self.full_data_dict, self.few_data_dict = self.load_data(root, train, dataset)

        print('full_data_num: %d' % count_data(self.full_data_dict))
        print('few_data_num: %d' % count_data(self.few_data_dict))

    def load_data(self, root, train, dataset):
        if dataset == 'cifar100':
            few_selected_label = random.Random(self.seed).sample(range(self.num_labels), self.nway)
            print('selected labeled', few_selected_label)

            full_data_dict = {}
            few_data_dict = {}

            d = CIFAR100(root, train=train, download=True)

            for i, (data, label) in enumerate(d):

                data = self.transform(data)

                if label in few_selected_label:
                    data_dict = few_data_dict
                else:
                    data_dict = full_data_dict

                if label not in data_dict:
                    data_dict[label] = [data]
                else:
                    data_dict[label].append(data)
            print(i + 1)
        else:
            raise NotImplementedError

        return full_data_dict, few_data_dict

    def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):
        if train:
            data_dict = self.full_data_dict
        else:
            data_dict = self.few_data_dict

        x = []
        label_y = [] # fake label: from 0 to (nway - 1)
        one_hot_y = [] # one hot for fake label
        class_y = [] # real label

        xi = []
        label_yi = []
        one_hot_yi = []
        

        map_label2class = []

        ### the format of x, label_y, one_hot_y, class_y is 
        ### [tensor, tensor, ..., tensor] len(label_y) = batch size
        ### the first dimension of tensor = num_shots

        for i in range(batch_size):

            # sample the class to train
            sampled_classes = random.sample(data_dict.keys(), nway)

            positive_class = random.randint(0, nway - 1)

            label2class = torch.LongTensor(nway)

            single_xi = []
            single_one_hot_yi = []
            single_label_yi = []
            single_class_yi = []


            for j, _class in enumerate(sampled_classes):
                if j == positive_class:
                    ### without loss of generality, we assume the 0th 
                    ### sampled  class is the target class
                    sampled_data = random.sample(data_dict[_class], num_shots+1)

                    x.append(sampled_data[0])
                    label_y.append(torch.LongTensor([j]))

                    one_hot = torch.zeros(nway)
                    one_hot[j] = 1.0
                    one_hot_y.append(one_hot)

                    class_y.append(torch.LongTensor([_class]))

                    shots_data = sampled_data[1:]
                else:
                    shots_data = random.sample(data_dict[_class], num_shots)

                single_xi += shots_data
                single_label_yi.append(torch.LongTensor([j]).repeat(num_shots))
                one_hot = torch.zeros(nway)
                one_hot[j] = 1.0
                single_one_hot_yi.append(one_hot.repeat(num_shots, 1))

                label2class[j] = _class

            shuffle_index = torch.randperm(num_shots*nway)
            xi.append(torch.stack(single_xi, dim=0)[shuffle_index])
            label_yi.append(torch.cat(single_label_yi, dim=0)[shuffle_index])
            one_hot_yi.append(torch.cat(single_one_hot_yi, dim=0)[shuffle_index])

            map_label2class.append(label2class)

        return [torch.stack(x, 0), torch.cat(label_y, 0), torch.stack(one_hot_y, 0), \
            torch.cat(class_y, 0), torch.stack(xi, 0), torch.stack(label_yi, 0), \
            torch.stack(one_hot_yi, 0), torch.stack(map_label2class, 0)]

    # def load_batch_data(self, train=True, batch_size=16, nway=5, num_shots=1):

    #     if train:
    #         data_dict = self.full_data_dict
    #     else:
    #         data_dict = self.few_data_dict

    #     x = torch.zeros(batch_size, self.input_channels, self.size, self.size)
    #     label_y = torch.LongTensor(batch_size).zero_()
    #     one_hot_y = torch.zeros(batch_size, nway)
    #     class_y = torch.LongTensor(batch_size).zero_()
    #     xi, label_yi, one_hot_yi, class_yi = [], [], [], []

    #     for i in range(nway*num_shots):
    #         xi.append(torch.zeros(batch_size, self.input_channels, self.size, self.size))
    #         label_yi.append(torch.LongTensor(batch_size).zero_())
    #         one_hot_yi.append(torch.zeros(batch_size, nway))
    #         class_yi.append(torch.LongTensor(batch_size).zero_())

    #     # sample data

    #     for i in range(batch_size):

    #         # sample the class to train
    #         sampled_classes = random.sample(data_dict.keys(), nway)

    #         positive_class = random.randint(0, nway - 1)

    #         indexes_perm = np.random.permutation(nway * num_shots)

    #         counter = 0

    #         for j, _class in enumerate(sampled_classes):
    #             if j == positive_class:
    #                 ### without loss of generality, we assume the 0th 
    #                 ### sampled  class is the target class
    #                 sampled_data = random.sample(data_dict[_class], num_shots+1)

    #                 x[i] = sampled_data[0]
    #                 label_y[i] = j

    #                 one_hot_y[i, j] = 1.0

    #                 class_y[i] = _class

    #                 shots_data = sampled_data[1:]
    #             else:
    #                 shots_data = random.sample(data_dict[_class], num_shots)

    #             for s_i in range(0, len(shots_data)):
    #                 xi[indexes_perm[counter]][i] = shots_data[s_i]
                    
    #                 label_yi[indexes_perm[counter]][i] = j
    #                 one_hot_yi[indexes_perm[counter]][i, j] = 1.0
    #                 class_yi[indexes_perm[counter]][i] = _class

    #                 counter += 1
    #     return [x, label_y, one_hot_y, class_y, torch.stack(xi, 1), torch.stack(label_yi, 1), \
    #         torch.stack(one_hot_yi, 1), torch.stack(class_yi, 1)]

    def load_tr_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(True, batch_size, nway, num_shots)

    def load_te_batch(self, batch_size=16, nway=5, num_shots=1):
        return self.load_batch_data(False, batch_size, nway, num_shots)

    def get_data_list(self, data_dict):
        data_list = []
        label_list = []
        for i in data_dict.keys():
            for data in data_dict[i]:
                data_list.append(data)
                label_list.append(i)

        now_time = time.time()

        random.Random(now_time).shuffle(data_list)
        random.Random(now_time).shuffle(label_list)

        return data_list, label_list

    def get_full_data_list(self):
        return self.get_data_list(self.full_data_dict)

    def get_few_data_list(self):
        return self.get_data_list(self.few_data_dict)

if __name__ == '__main__':
    D = self_DataLoader('/home/lab5300/Data', True)

    [x, label_y, one_hot_y, class_y, xi, label_yi, one_hot_yi, class_yi] = \
        D.load_tr_batch(batch_size=16, nway=5, num_shots=5)
    print(x.size(), label_y.size(), one_hot_y.size(), class_y.size())
    print(xi.size(), label_yi.size(), one_hot_yi.size(), class_yi.size())

    # print(label_y)
    # print(one_hot_y)

    print(label_yi[0])
    print(one_hot_yi[0])