from torchvision.models import  resnet18
from networks import GeomCNN, TripletNet,ClassificationNet,DoubleNet
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm 
import csv 
from PIL import Image 
from torchvision.transforms.functional import pil_to_tensor
import os 
import sys
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import train_test_split,KFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import numpy as np 
import cornet 

class TestImageNet(Dataset):
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
        return img,[(self.all_images[idx]['category'],self.all_images[idx]['class'])]

if __name__=='__main__':
    cond=sys.argv[1]
    run=int(sys.argv[2])
    data=sys.argv[3]

    if data=='train':
        dset=TestImageNet(mode='train')
    else:
        dset=TestImageNet(mode='test')
    if cond=='baseline':
        path='saved_imagenet_models/run_{}_baseline_weights.pt'.format(run)
    else:
        path='saved_imagenet_models/run_{}_triplet_weights.pt'.format(run)
    e_net=resnet18(weights=None) 
    
    if cond=='triplet':
        full_model = TripletNet(e_net)
    elif cond=='baseline':
        full_model = DoubleNet(e_net)
    full_model.load_state_dict(torch.load(path,map_location='cpu'))
    embedding_net=full_model.embedding_net

    categories=[]
    embeddings=[]
    #embedding_net.cuda()
    loader = torch.utils.data.DataLoader(dset, batch_size=128, shuffle=False, num_workers=4)
    for batch,labels in tqdm(loader):
        batch=batch.cuda()
        embedding=embedding_net(batch)
        embeddings.append(embedding.detach().cpu().numpy())
        categories+=labels[0][0]
        #del batch 
        #torch.cuda.empty_cache()
    for i in range(len(categories)): 
        categories[i]=dset.category_list.index(categories[i])
    categories=np.asarray(categories)
    embeddings=np.vstack(embeddings)


    X=embeddings.copy()
    y=categories 
    acc=[]
    kf=KFold(n_splits=3,shuffle=True) 
    for train_index,test_index in kf.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        reg_model=SGDClassifier(verbose=True)
        reg_model.fit(X_train,y_train)
        preds=reg_model.predict(X_test)
        acc.append(accuracy_score(y_test,preds))
    acc=np.asarray(acc)


    np.save('imagenet_results/{}_{}_{}.npy'.format(cond,run,data),acc)





    
