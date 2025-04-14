import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from trainer import fit
import numpy as np
from tqdm import tqdm 
import numpy as np 
from networks import GeomCNN, TripletNet,ClassificationNet,DoubleNet
from losses import TripletLoss,InfoNCE
from torch.utils.data import Dataset
from torchvision.io import read_image,ImageReadMode
import pickle 
from sklearn.linear_model import LogisticRegressionCV,RidgeCV,LinearRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import KFold
from PIL import Image 
from metrics import AccumulatedAccuracyMetric
import torch 
import torch.nn as nn 
import sys 
from torchvision.models import  resnet18
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import random
import numpy as np
from tqdm import tqdm 
import csv 
from torchvision.transforms.functional import pil_to_tensor
import os 
class TripletImageNet(Dataset):
    def __init__(self,mode='train',level_prob=0.8,debug=False):
        self.debug=debug 
        self.level_prob=level_prob 
        #PUT YOUR IMAGENET FOLDER HERE. The filepath was taken out for anonymitiy. 
        self.IMAGENET_DIR='/path_to_imagenet/'
        filename = mode + '_imgs.csv'
        self.category_to_class={}
        self.class_to_category={}
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            category_list = []
            class_list = []
            for row in csv_reader:
                if row[1] not in self.category_to_class.keys():
                    self.category_to_class[row[1]]=[]
                self.category_to_class[row[1]]=row[0]
                assert row[0] not in self.class_to_category.keys()
                self.class_to_category[row[0]]=row[1]
                category_list.append(row[1])
                class_list.append(row[0])
        

        self.class_list=list(set(self.class_to_category.keys()))
        self.category_list=list(set(self.category_to_class.keys()))

        self.resize_transform=transforms.Resize((224,224)) 
        self.class_to_image={}
        self.category_to_image={}
        self.image_to_rest={}

        self.all_images=[]
        for c in tqdm(self.class_list):
            if c not in self.class_to_image.keys():
                self.class_to_image[c]=[]

            cat=self.class_to_category[c]
            if cat not in self.category_to_image.keys():
                self.category_to_image[cat]=[]

            c_dir=self.IMAGENET_DIR+c+'/'
            if os.path.exists(c_dir):
                files=os.listdir(c_dir)
                for f in files:
                    self.all_images.append({
                        'path':c_dir+f,
                        'category':cat,
                        'class':c 
                    })
                    self.image_to_rest[c_dir+f]=[cat,c]
                    self.class_to_image[c].append(c_dir+f)
                    self.category_to_image[cat].append(c_dir+f)
                
    def load_img(self,path):
        raw_img=pil_to_tensor(Image.open(path).convert('RGB'))
        I=self.resize_transform(raw_img)/255.0
        if I.shape!=(3,224,224):
            print(I.shape)
            print(path) 
        assert I.shape==(3,224,224)
        return I 
    
    def __len__(self):
        return len(self.all_images)

    def __getitem__(self,index):
        anchor_image=self.load_img(self.all_images[index]['path'])
        anchor_category=self.all_images[index]['category']
        anchor_class=self.all_images[index]['class']
        if random.random()<self.level_prob: #High-level, positive same category independent class
            if self.debug:
                print("High-level Triplet, Category=",anchor_category)
            pos_img_path=np.random.choice(self.category_to_image[anchor_category])
            pos_cat,pos_class=self.image_to_rest[pos_img_path]
            assert pos_cat==anchor_category
            positive_image=self.load_img(pos_img_path)
            neg_img_path=np.random.choice(self.all_images)['path']
            neg_cat,neg_class=self.image_to_rest[neg_img_path]
            negative_image=self.load_img(neg_img_path) #Negative image for high-level triplet can be from any category. 
        else: #Low-level, positive same category and class 
            if self.debug:
                print("Low-level Triplet, Class=",anchor_class)
            pos_img_path=np.random.choice(self.class_to_image[anchor_class])
            pos_cat,pos_class=self.image_to_rest[pos_img_path]
            assert pos_cat==anchor_category
            assert pos_class==anchor_class 
            positive_image=self.load_img(pos_img_path)
            neg_img_path=np.random.choice(self.category_to_image[anchor_category])
            neg_cat,neg_class=self.image_to_rest[neg_img_path]
            assert neg_cat==anchor_category 
            negative_image=self.load_img(neg_img_path) #Negative image, same category but any class. 
        #return (anchor_image,positive_image,negative_image),[(anchor_category,anchor_class),(pos_cat,pos_class),(neg_cat,neg_class)]
        return (anchor_image,positive_image,negative_image),[]


    

class NCEImageNet(Dataset):
    def __init__(self,mode='train'):
        #PUT YOUR IMAGENET FOLDER HERE. The filepath was taken out for anonymitiy. 
        self.IMAGENET_DIR='/path_to_imagenet/'
        filename = mode + '_imgs.csv'
        self.category_to_class={}
        self.class_to_category={}
        with open(filename) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            category_list = []
            class_list = []
            for row in csv_reader: 
                if row[1] not in self.category_to_class.keys():
                    self.category_to_class[row[1]]=[]
                self.category_to_class[row[1]]=row[0]
                assert row[0] not in self.class_to_category.keys()
                self.class_to_category[row[0]]=row[1]
                category_list.append(row[1])
                class_list.append(row[0])
        

        self.class_list=list(set(self.class_to_category.keys()))
        self.category_list=list(set(self.category_to_class.keys()))

        self.resize_transform=transforms.Resize((224,224)) 
        self.class_to_image={}
        self.category_to_image={}
        self.image_to_rest={}

        self.all_images=[]
        for c in tqdm(self.class_list):
            if c not in self.class_to_image.keys():
                self.class_to_image[c]=[]

            cat=self.class_to_category[c]
            if cat not in self.category_to_image.keys():
                self.category_to_image[cat]=[]

            c_dir=self.IMAGENET_DIR+c+'/'
            if os.path.exists(c_dir):
                files=os.listdir(c_dir)
                for f in files:
                    self.all_images.append({
                        'path':c_dir+f,
                        'category':cat,
                        'class':c 
                    })
                    self.image_to_rest[c_dir+f]=[cat,c]
                    self.class_to_image[c].append(c_dir+f)
                    self.category_to_image[cat].append(c_dir+f)
        s=1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.aug_transforms=transforms.Compose([transforms.RandomResizedCrop(size=224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.GaussianBlur(11),
                                              transforms.RandomGrayscale(p=0.2)])
    def load_img(self,path):
        raw_img=pil_to_tensor(Image.open(path).convert('RGB'))
        I=self.resize_transform(raw_img)/255.0
        if I.shape!=(3,224,224):
            print(I.shape)
            print(path) 
        assert I.shape==(3,224,224)
        return I 

    def __len__(self):
        return len(self.all_images)
    def __getitem__(self,idx):
        img=self.load_img(self.all_images[idx]['path'])
        aug=self.aug_transforms(img)
        #return (img,aug),[(self.all_images[idx]['category'],self.all_images[idx]['class'])]
        return (img,aug),[]
        


def train_triplet_network(train_dataset,test_dataset,run,cond): 
    save_dir='saved_imagenet_models/'+cond+'_run_'+str(run)+"/"
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    cuda = torch.cuda.is_available()
    batch_size = 256
    kwargs = {'num_workers': 8} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    margin = 1.
    embedding_net=resnet18()   
    triplet_model = TripletNet(embedding_net)
    if cuda:
        triplet_model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(triplet_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10
    log_interval = 50
    fit(triplet_train_loader, triplet_test_loader, triplet_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,save_dir)
    return triplet_model.state_dict()

def train_baseline_network(train_dataset,test_dataset,run):
    save_dir='saved_imagenet_models/baseline_run_'+str(run)+"/"
    if not(os.path.exists(save_dir)):
        os.mkdir(save_dir)
    cuda = torch.cuda.is_available()
    batch_size = 256
    kwargs = {'num_workers': 8} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    embedding_net2=resnet18(weights=None)
    baseline_model = DoubleNet(embedding_net2)
    if cuda:
        baseline_model.cuda() 
    loss_fn = InfoNCE()
    lr = 1e-3
    optimizer = optim.Adam(baseline_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 10
    log_interval = 50
    fit(train_loader, test_loader, baseline_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,save_dir)
    return baseline_model.state_dict()





if __name__=='__main__':
    run=int(sys.argv[1])
    cond=sys.argv[2]
    triplet_train_dataset=TripletImageNet(mode='train',level_prob=0.8)
    triplet_test_dataset=TripletImageNet(mode='test')
    baseline_train_dataset=NCEImageNet()
    baseline_test_dataset=NCEImageNet(mode='test')
    triplet_acc=[]
    baseline_acc=[]
    if cond=='triplet':
        weights=train_triplet_network(triplet_train_dataset,triplet_test_dataset,run,cond)
        torch.save(weights,'saved_imagenet_models/run_{}_{}_weights.pt'.format(run,cond))
    elif cond=='baseline':
        weights=train_baseline_network(baseline_train_dataset,baseline_test_dataset,run) 
        torch.save(weights,'saved_imagenet_models/run_{}_baseline_weights.pt'.format(run))
   
