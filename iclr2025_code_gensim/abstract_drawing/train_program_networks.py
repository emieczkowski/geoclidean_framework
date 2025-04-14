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
from datasets import TripletGeom,NCEGeom
from networks import GeomCNN, TripletNet,DoubleNet
from losses import TripletLoss,InfoNCE
from torch.utils.data import Dataset
from torchvision.io import read_image,ImageReadMode
import pickle 
from sklearn.linear_model import LogisticRegressionCV,RidgeCV,LinearRegression
from sklearn.metrics import accuracy_score,r2_score
from sklearn.model_selection import KFold
import torch 
import torch.nn as nn 
import cornet 
import sys 
     
class TestGeom(Dataset):
    def __init__(self,reg_df,mode=ImageReadMode.GRAY):
        self.images=[]
        for i in range(len(reg_df)):
            if reg_df.iloc[i]['grammar']=='celtic':
                self.images.append(read_image('celtic_images/{}.png'.format(reg_df.iloc[i]['id']),mode))
            else:
                self.images.append(read_image('greek_images/{}.png'.format(reg_df.iloc[i]['id']),mode)) 
    
    def __getitem__(self,index):
        return self.images[index]/255.0

    def __len__(self):
        return len(self.images)
    
def extract_embeddings(dataloader, model):
    cuda = torch.cuda.is_available()
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 256))
        k = 0
        for images in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.embedding_net(images).data.cpu().numpy()
            k += len(images)
    return embeddings

def run_classification(dataloader,model):
    embeddings=extract_embeddings(dataloader,model)
    kf=KFold(n_splits=5,shuffle=True)
    X=embeddings.copy()
    reg_df=pickle.load(open('regression_df.pkl','rb'))
    y=np.asarray([int(g=='greek') for g in np.asarray(reg_df['grammar'])])
    acc=[]
    for train_index,test_index in kf.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        reg_model=LogisticRegressionCV()
        reg_model.fit(X_train,y_train)
        preds=reg_model.predict(X_test)
        acc.append(accuracy_score(y_test,preds))
    return np.asarray(acc)

def run_regression(dataloader,model,control_vector):
    embeddings=extract_embeddings(dataloader,model)
    kf=KFold(n_splits=5,shuffle=True)
    X=embeddings.copy()
    reg_df=pickle.load(open('regression_df2.pkl','rb')) # same as reg_df but has the extra column with the program primitives. 
    reg_df['number_prims']=[sum(x) for x in list(reg_df['vec2'])]
    y=np.asarray(reg_df['number_prims'])
    control_model=LinearRegression()
    control_model.fit(control_vector,y)
    y=y-control_model.predict(control_vector)
    y=np.asarray([int(g=='greek') for g in np.asarray(reg_df['grammar'])])

    acc=[]
    for train_index,test_index in kf.split(X):
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        reg_model=RidgeCV()
        reg_model.fit(X_train,y_train)
        preds=reg_model.predict(X_test)
        score=r2_score(y_test,preds)
        acc.append(score)
        print(score)
    return np.asarray(acc)


    

def train_triplet_network(train_dataset,test_dataset,test2_dataset):
    control_vec=np.asarray([torch.mean(im).item() for im in test2_dataset]).reshape((-1,1))
    cuda = torch.cuda.is_available()
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    margin = 1.
    embedding_net = GeomCNN()
    triplet_model = TripletNet(embedding_net)
    if cuda:
        triplet_model.cuda()
    loss_fn = TripletLoss(margin)
    lr = 1e-3
    optimizer = optim.Adam(triplet_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 12
    log_interval = 100
    fit(triplet_train_loader, triplet_test_loader, triplet_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    test_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    #acc=run_regression(test_loader,triplet_model)
    #r2=run_regression2(test_loader,triplet_model,control_vec)
    acc=run_classification(test_loader,triplet_model)
    r2=run_regression(test_loader,triplet_model,control_vec)
    #r2_2=run_regression4(test_loader,triplet_model,control_vec)
    #print(acc,r2)
    #return acc,r2
    return acc,r2

def train_baseline_network(train_dataset,test_dataset,test2_dataset):
    control_vec=np.asarray([torch.mean(im).item() for im in test2_dataset]).reshape((-1,1))
    cuda = torch.cuda.is_available()
    batch_size = 128
    kwargs = {'num_workers': 1, } if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    #embedding_net2 = GeomCNN()
    embedding_net2 = GeomCNN()
    baseline_model = DoubleNet(embedding_net2)
    if cuda:
        baseline_model.cuda()
    loss_fn = InfoNCE()
    lr = 1e-3
    optimizer = optim.Adam(baseline_model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 12
    log_interval = 50
    fit(train_loader, test_loader, baseline_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    test_loader = torch.utils.data.DataLoader(test2_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    #return run_regression2(test_loader,baseline_model,control_vec)
    acc=run_classification(test_loader,baseline_model)
    r2=run_regression(test_loader,baseline_model,control_vec)
    return acc,r2 






if __name__=='__main__':
    net_type='GeomCNN'
    mode=ImageReadMode.GRAY
    run=int(sys.argv[2])
    save_name=net_type 
    triplet_train_dataset=TripletGeom(train=True,mode=mode)
    triplet_test_dataset=TripletGeom(train=False,mode=mode)
    baseline_train_dataset=NCEGeom(train=True,mode=mode)
    baseline_test_dataset=NCEGeom(train=False,mode=mode)
    reg_df=pickle.load(open('regression_df.pkl','rb'))
    test2_dataset=TestGeom(reg_df,mode=mode)
    triplet_acc=[]
    baseline_acc=[]
    print("RUN",run)
    triplet_acc.append(train_triplet_network(triplet_train_dataset,triplet_test_dataset,test2_dataset))
    baseline_acc.append(train_baseline_network(baseline_train_dataset,baseline_test_dataset,test2_dataset))
    triplet_acc=np.asarray(triplet_acc)
    baseline_acc=np.asarray(baseline_acc)


