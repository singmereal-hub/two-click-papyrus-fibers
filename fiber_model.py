"""Small endpoint-conditioned likelihood model for tracing papyrus fibers in CT."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b):
    return nn.Sequential(nn.Conv3d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),
                         nn.Conv3d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())


class FiberLikelihood(nn.Module):
    """Predict a distribution over the target fiber's location in each normal plane."""
    def __init__(self):
        super().__init__()
        self.first=block(3,12)
        self.middle=block(12,20)
        self.deep=block(20,32)
        self.up1=block(52,20)
        self.up2=block(32,12)
        self.final=nn.Conv3d(14,1,1)

    def forward(self,x):
        _,_,t,h,w=x.shape
        xx=torch.arange(w,device=x.device,dtype=x.dtype)-(w-1)/2
        yy=torch.arange(h,device=x.device,dtype=x.dtype)-(h-1)/2
        radial=(xx.square()[None,:]+yy.square()[:,None]).sqrt()
        prior=(radial/12).clamp(max=2)[None,None,None].expand_as(x)
        local=x-F.avg_pool3d(x,(3,7,7),stride=1,padding=(1,3,3),count_include_pad=False)
        hi=self.first(torch.cat([x,local*3,prior],dim=1))
        mid=self.middle(F.avg_pool3d(hi,2,ceil_mode=True))
        deep=self.deep(F.avg_pool3d(mid,2,ceil_mode=True))
        mid=self.up1(torch.cat([mid,F.interpolate(deep,size=mid.shape[-3:],mode="trilinear",align_corners=False)],dim=1))
        feat=self.up2(torch.cat([hi,F.interpolate(mid,size=hi.shape[-3:],mode="trilinear",align_corners=False)],dim=1))
        cy,cx=h//2,w//2
        left=feat[:,:,:1,cy-1:cy+2,cx-1:cx+2].mean(dim=(3,4),keepdim=True)
        right=feat[:,:,-1:,cy-1:cy+2,cx-1:cx+2].mean(dim=(3,4),keepdim=True)
        desc=F.normalize(feat,dim=1,eps=1e-4)
        a=(desc*F.normalize(left,dim=1,eps=1e-4)).sum(dim=1,keepdim=True)
        b=(desc*F.normalize(right,dim=1,eps=1e-4)).sum(dim=1,keepdim=True)
        tau=torch.linspace(0,1,t,device=x.device,dtype=x.dtype)[None,None,:,None,None]
        return self.final(torch.cat([feat,(1-tau)*a+tau*b,torch.minimum(a,b)],dim=1))[:,0]


class Predictor:
    def __init__(self, path: Path, device=None):
        self.device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net=FiberLikelihood().to(self.device)
        checkpoint=torch.load(path,map_location="cpu",weights_only=True)
        self.net.load_state_dict(checkpoint["state"])
        self.net.eval()

    @torch.inference_mode()
    def logits(self,images):
        x=torch.from_numpy(np.ascontiguousarray(images)).to(self.device,dtype=torch.float32)[None,None]/127.5-1.
        return self.net(x)[0].cpu().numpy()
