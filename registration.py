"""Two-click tracing by joint registration of short and longer CT slice intervals."""
from __future__ import annotations

import cv2
import numpy as np


def lookup(fields, points):
    n,h,w,_=fields.shape
    x=np.clip(points[:,0],0,w-1.001)
    y=np.clip(points[:,1],0,h-1.001)
    xx=x.astype(int); yy=y.astype(int)
    dx=(x-xx)[:,None]; dy=(y-yy)[:,None]; z=np.arange(n)
    return ((1-dy)*((1-dx)*fields[z,yy,xx]+dx*fields[z,yy,xx+1])+
            dy*((1-dx)*fields[z,yy+1,xx]+dx*fields[z,yy+1,xx+1]))


def registrations(images, forward, reverse, lengths=(1,2,4)):
    count=len(images)
    all_f=[];all_r=[];starts=[];ends=[]
    for step in lengths:
        if step>=count:
            continue
        if step==1:
            f,r=forward,reverse
        else:
            def obtain(a,b):
                return cv2.calcOpticalFlowFarneback(a,b,None,.5,2,13,3,5,1.1,0)
            f=np.stack([obtain(a,b) for a,b in zip(images[:-step],images[step:])])
            r=np.stack([obtain(b,a) for a,b in zip(images[:-step],images[step:])])
        all_f.append(f); all_r.append(r)
        starts.extend(range(count-step)); ends.extend(range(step,count))
    return np.array(starts),np.array(ends),np.concatenate(all_f),np.concatenate(all_r)


def local_centers(path,score):
    h,w=score.shape[1:]
    offsets=np.arange(-4,5)
    dy,dx=np.meshgrid(offsets,offsets,indexing="ij")
    x=np.clip(np.rint(path[:,0,None,None]).astype(int)+dx,0,w-1)
    y=np.clip(np.rint(path[:,1,None,None]).astype(int)+dy,0,h-1)
    z=np.arange(len(path))[:,None,None]
    logits=score[z,y,x]-((x-path[:,0,None,None])**2+(y-path[:,1,None,None])**2)/(2*1.8**2)
    weights=np.exp(logits-logits.max(axis=(1,2),keepdims=True))
    weights/=weights.sum(axis=(1,2),keepdims=True)
    return np.column_stack([(weights*x).sum((1,2)),(weights*y).sum((1,2))])


def solve_bridge(initial,edges, score=None, attraction=0.0):
    """Robust graph least squares on material displacement; both clicked ends are fixed."""
    start,end,fwd,bwd=edges
    n=len(initial)
    B=np.zeros((len(start),n))
    B[np.arange(len(start)),start]=-1.
    B[np.arange(len(start)),end]=1.
    second=np.diff(np.eye(n),n=2,axis=0)
    curvature=.24*(second.T@second)
    anchor=np.asarray(initial[[0,-1]],dtype=float)
    current=initial.astype(float).copy()
    for _ in range(7):
        f=lookup(fwd,current[start]); r=lookup(bwd,current[end]); delta=(f-r)*.5
        closure=(np.linalg.norm(f+lookup(bwd,current[start]+f),axis=1)+
                 np.linalg.norm(r+lookup(fwd,current[end]+r),axis=1))*.5
        residual=np.linalg.norm((current[end]-current[start])-delta,axis=1)
        weights=(end-start)**(-.8)/(1+(closure/1.5)**2) / np.maximum(1,residual/2)
        target=local_centers(current,score) if score is not None else current
        matrix = B.T@(weights[:,None]*B)+curvature+(.012+attraction)*np.eye(n)
        rhs=B.T@(weights[:,None]*delta)+.012*initial+attraction*target
        rhs=rhs[1:-1]-matrix[1:-1][:,[0,-1]]@anchor
        new=np.linalg.solve(matrix[1:-1,1:-1],rhs)
        current[1:-1]=.25*current[1:-1]+.75*new
        current=np.clip(current,0,fwd.shape[1]-1)
    return current.astype(np.float32)
