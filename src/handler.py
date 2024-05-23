import torch
import numpy as np
from .losses import BinaryFocalLoss
from .dataset import MultilabelDataset
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

class TrainValHandler():
    def __init__(self,model,device,trainset,valset,save_path,
                 alpha=None,gamma=0,ohem=False,
                 lr=1e-3,epochs=90,patience=50,batchsize=1024,nbatches=2) -> None:
        self.model = model
        self.device = device
        self.path = save_path
        self.trainset = MultilabelDataset(trainset)
        self.valset = MultilabelDataset(valset)

        self.loss = BinaryFocalLoss(alpha,gamma,reduction=not ohem)
        self.optim = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=30,gamma=0.1)
        self.epochs = epochs
        self.patience = patience
        self.min_loss = torch.inf
        self.ohem = ohem
        self.batchsize = batchsize
        self.nbatches = nbatches

        try:
            workers = cpu_count()
        except:
            workers = 1
        if ohem:
            self.traindataloader = DataLoader(self.trainset,self.batchsize*self.nbatches,shuffle=True,num_workers=workers)
        else:
            self.traindataloader = DataLoader(self.trainset,self.batchsize,shuffle=True,num_workers=workers)
        self.valdataloader = DataLoader(self.valset,self.batchsize,shuffle=True,num_workers=workers)
    
    def save_model(self):
        path = self.path
        torch.save(self.model.state_dict(),path)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        for i, samples in enumerate(self.traindataloader):
            x,y = samples
            x = x.to(self.device,dtype=torch.float)
            y = y.to(self.device,dtype=torch.float)
            pred = self.model(x)
            if self.ohem:
                loss,_ = torch.topk(self.loss(pred,y),pred.shape[0]//self.nbatches)
                loss = torch.mean(loss)
            else:
                loss = self.loss(pred,y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            print('batch %d/%d, training loss %.6f' % (i+1,len(self.traindataloader),train_loss/(i+1)),end='\r')
        return train_loss/len(self.traindataloader)

    def val_one_epoch(self):
        self.model.eval()
        val_loss = 0
        for i, samples in enumerate(self.valdataloader):
            x,y = samples
            x = x.to(self.device,dtype=torch.float)
            y = y.to(self.device,dtype=torch.float)
            with torch.no_grad():
                pred = self.model(x)
                loss = self.loss(pred,y)
                if self.ohem:
                    loss = torch.mean(loss)
                val_loss += loss.item()
        return val_loss/len(self.valdataloader)

    def train(self):
        self.model.to(self.device)
        patience = 0
        history = {
            "training loss":[],
            "validation loss":[]
        }
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.val_one_epoch()
            self.scheduler.step()
            history["training loss"].append(train_loss)
            history["validation loss"].append(val_loss)
            print("epoch %d/%d, training loss: %.6f, validation loss: %.6f"%(epoch+1,self.epochs,train_loss,val_loss))
            if val_loss < self.min_loss:
                self.min_loss = val_loss
                patience = 0
                self.save_model()
                print("save best model at epch %d"%(epoch+1))
            else:
                patience += 1

            if patience == self.patience:
                print("no improvement from last %d epoch, stop training"%patience)
                break
        return history

class TestHandler():
    def __init__(self,model,device,testset,batchsize,
                 labels = ["WiFi", "LTE", "Zigbee", "LoRa", "BLE"]) -> None:
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.testset = MultilabelDataset(testset)
        try:
            workers = cpu_count()
        except:
            workers = 1
        self.testdataloader = DataLoader(self.testset,batchsize,shuffle=True,num_workers=workers)
        self.labels = labels

    def eval_one_step(self,i):
        x,y = self.testset[i]
        x = x.reshape((1,)+x.shape)
        y = y.reshape((1,)+y.shape)
        x = x.to(self.device,dtype=torch.float)
        y = y.to(self.device,dtype=torch.float)
        with torch.no_grad():
            y_pred = self.model(x)
        
        y_pred[y_pred>=0.5] = 1
        y_pred[y_pred<0.5] = 0

        return y, y_pred

    def get_plot(self,y,plotname):
        if self.device != "cpu":
            data = y.cpu()
        fig,axes = plt.subplots(data.shape[1],sharex=True,figsize=(8,6))
        for i, ax in enumerate(axes):
            ax.imshow(data[:,i,:].reshape(1,-1),cmap='cividis',aspect='auto',extent=[-12.5,12.5,1,0])
            ax.set_ylabel(self.labels[i],fontsize=12)
            ax.set_yticks([])

        fig.supxlabel("Frequency (MHz)",fontsize=18)
        fig.savefig(plotname,format='eps')

    def display(self,i):
        y, y_pred = self.eval_one_step(i)
        self.get_plot(y_pred,"predicted%d"%i)
        self.get_plot(y,"groundtruth%d"%i)

    def compute_iou(self,intersec,union):
        iou = intersec/union
        return torch.nan_to_num(iou)

    def compute_recall(self,intersec,area_t):
        recall = intersec/area_t
        return torch.nan_to_num(recall)

    def compute_precision(self,intersec,area_p):
        precision = intersec/area_p
        return torch.nan_to_num(precision)

    def evaluate(self):
        intersec = []
        union = []
        area_t = []
        area_p = []
        for i, samples in enumerate(self.testdataloader):
            x,y = samples
            x = x.to(self.device,dtype=torch.float)
            y = y.to(self.device,dtype=torch.float)
            with torch.no_grad():
                y_pred = self.model(x)
            y_pred[y_pred>=0.5] = 1
            y_pred[y_pred<0.5] = 0

            intersec_batch = torch.sum(torch.logical_and(y,y_pred),(0,2))
            union_batch = torch.sum(torch.logical_or(y,y_pred),(0,2))
            area_true_batch = torch.sum(y,(0,2))
            area_pred_batch = torch.sum(y_pred,(0,2))

            intersec.append(intersec_batch)
            union.append(union_batch)
            area_t.append(area_true_batch)
            area_p.append(area_pred_batch)

            mean_iou = torch.mean(self.compute_iou(intersec_batch,union_batch))
            print("batch %d/%d, mean iou: %.6f"%(i+1,len(self.testdataloader),mean_iou),end='\r')
        intersec = torch.sum(torch.stack(intersec,0),dim=0)
        union = torch.sum(torch.stack(union,0),dim=0)
        area_t = torch.sum(torch.stack(area_t,0),dim=0)
        area_p = torch.sum(torch.stack(area_p,0),dim=0)
        iou = self.compute_iou(intersec,union)
        recall = self.compute_recall(intersec,area_t)
        precision = self.compute_precision(intersec,area_p)
        return iou.cpu().numpy(), recall.cpu().numpy(), precision.cpu().numpy()

