import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision.io import read_image,ImageReadMode
from tqdm import tqdm 
import torch 
from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur

class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class TripletGeom(Dataset):
    def __init__(self,celtic_loc='celtic_images/',greek_loc='greek_images/',train=True,mode=ImageReadMode.GRAY):
        if train:
            size=20000
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
            self.dset_size=size 
        else:
            size=400
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
            self.dset_size=size 
        self.train=train 

    def __getitem__(self,index):
        if index<self.dset_size:
            anchor_grammar=0 #0=celtic, 1=greek
            anchor_idx=index 
        else:
            anchor_grammar=1
            anchor_idx=index-self.dset_size
        positive_idx=np.random.choice(np.arange(self.dset_size))
        negative_grammar=np.random.choice([0,1])
        negative_idx=np.random.choice(np.arange(self.dset_size))

        if anchor_grammar==0:
            img1=self.celtic_data[anchor_idx]/255.0
            img2=self.celtic_data[positive_idx]/255.0
        else:
            img1=self.greek_data[anchor_idx]/255.0
            img2=self.greek_data[positive_idx]/255.0
        if negative_grammar==0:
            img3=self.celtic_data[negative_idx]/255.0
        else:
            img3=self.greek_data[negative_idx]/255.0
        return (img1,img2,img3),[]
    def __len__(self):
        return self.dset_size*2 

class Geom(Dataset):
    def __init__(self,celtic_loc='celtic_images/',greek_loc='greek_images/',train=True,mode=ImageReadMode.GRAY):
        if train:
            size=20000
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
        else:
            size=400
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
        
        self.all_data=self.celtic_data+self.greek_data
        self.all_labels=[0 for _ in range(len(self.celtic_data))]+[1 for _ in range(len(self.greek_data))]
        self.train=train 
    
    def __getitem__(self,index):
        img=self.all_data[index]/255.0 
        lbl=self.all_labels[index]
        return img,lbl 

    def __len__(self):
        return len(self.all_data)

class NCEGeom(Dataset):
    def __init__(self,celtic_loc='celtic_images/',greek_loc='greek_images/',train=True,mode=ImageReadMode.GRAY):
        if train:
            size=20000
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(size))]
        else:
            size=400
            self.celtic_data=[read_image('celtic_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
            self.greek_data=[read_image('greek_images/{}.png'.format(i),mode) for i in tqdm(range(20000,20000+size))]
        
        self.all_data=self.celtic_data+self.greek_data
        self.train=train 
        size=self.celtic_data[0].shape[1]
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.GaussianBlur(11)])
        
    
    def __getitem__(self,index):
        img=self.all_data[index]/255.0 
        aug=self.data_transforms(img)
        return (img,aug ),[]

    def __len__(self):
        return len(self.all_data)








        





class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
