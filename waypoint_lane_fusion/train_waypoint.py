#!/usr/bin/env python3
"""Train ResNet18 to regress x,y and estimate confidence from prediction error."""
import argparse, csv, json, random
from pathlib import Path
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


class WaypointDataset(Dataset):
    def __init__(self, csv_path, augment=True):
        self.root=Path(csv_path).resolve().parent; self.rows=list(csv.DictReader(Path(csv_path).open())); self.augment=augment
        ops=[transforms.Resize((224,224))]
        if augment: ops.append(transforms.ColorJitter(.25,.25,.2,.05))
        ops += [transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]
        self.tf=transforms.Compose(ops)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; image=self.tf(Image.open(self.root/r["image"]).convert("RGB"))
        target=torch.tensor([float(r["x"]),float(r["y"])],dtype=torch.float32)
        if self.augment and random.random() < .5:
            image=torch.flip(image,dims=[2]); target[0]=1.0-target[0]
        return image,target


class LaneNet(nn.Module):
    def __init__(self):
        super().__init__(); self.backbone=models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,3))
    def forward(self,x):
        out=self.backbone(x); xy=torch.sigmoid(out[:,:2]); confidence=torch.sigmoid(out[:,2:3]); return torch.cat([xy,confidence],1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("csv"); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch",type=int,default=64); p.add_argument("--output",default="artifacts/lane_resnet18.pt"); p.add_argument("--seed",type=int,default=42); p.add_argument("--workers",type=int,default=0,help="DataLoader workers; keep 0 in low-/dev-shm containers"); a=p.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    full=WaypointDataset(a.csv,augment=True); n=max(1,int(.15*len(full))); train_n=len(full)-n
    train_indices,val_indices=random_split(range(len(full)),[train_n,n],generator=torch.Generator().manual_seed(a.seed))
    train=torch.utils.data.Subset(full,train_indices.indices)
    validation_full=WaypointDataset(a.csv,augment=False); val=torch.utils.data.Subset(validation_full,val_indices.indices)
    device="cuda" if torch.cuda.is_available() else "cpu"; model=LaneNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4); loss_fn=nn.SmoothL1Loss()
    history=[]; best=float("inf"); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        for x,y in DataLoader(train,a.batch,shuffle=True,num_workers=a.workers):
            x,y=x.to(device),y.to(device); out=model(x); xy_loss=loss_fn(out[:,:2],y)
            target_conf=torch.exp(-8*torch.norm(out[:,:2].detach()-y,dim=1,keepdim=True)); loss=xy_loss+.15*nn.functional.binary_cross_entropy(out[:,2:3],target_conf)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); errors=[]
        with torch.no_grad():
            for x,y in DataLoader(val,a.batch):
                prediction=model(x.to(device))[:,:2].cpu(); errors.extend(torch.norm(prediction-y,dim=1).tolist())
        val_error=sum(errors)/max(1,len(errors)); history.append({"epoch":epoch+1,"val_normalized_l2":val_error})
        print("epoch %d train_loss %.5f val_l2 %.5f"%(epoch+1,float(loss),val_error))
        if val_error < best:
            best=val_error; torch.jit.script(model.cpu().eval()).save(str(output)); model.to(device)
    metrics={"samples":len(full),"train_samples":train_n,"val_samples":n,"best_val_normalized_l2":best,"history":history}
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("best model",output,"metrics",output.with_suffix(".metrics.json"))


if __name__=="__main__": main()
