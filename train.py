"""Train on Scroll 1 only. Scroll 5 is never opened by this script."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataset import tasks_for_cube
from fiber_trace import make_tube, read_volume
from fiber_model import FiberLikelihood


def make_cache(cache: Path, data: Path) -> None:
    images, labels, cubes = [],[],[]
    for idx,path in enumerate(sorted(data.glob("s1_*.tif"))):
        vol=read_volume(path)
        tasks=tasks_for_cube(path,0,per_tree=8)
        def sample(task):
            ref=task.path
            a,b=ref[0],ref[-1]
            axis=b-a
            t=(ref-a)@axis / (axis@axis)
            if np.any(np.diff(t)<-.06):
                return None
            tube=make_tube(vol,a,b,planes=33)
            order=np.argsort(t)
            t=t[order]
            grid=np.linspace(0,1,33)
            lateral=np.column_stack([np.interp(grid,t,((ref-a)@v)[order]) for v in (tube.u,tube.v)])
            return tube.images,(lateral+tube.radius).astype(np.float32)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for item in ex.map(sample,tasks):
                if item is not None:
                    x,y=item
                    images.append(x)
                    labels.append(y)
                    cubes.append(idx)
        print("cache",path.name,len(tasks),flush=True)
    if not images: raise ValueError("No Scroll 1 training data")
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez(cache,x=np.stack(images),y=np.stack(labels),cube=np.array(cubes,dtype=np.uint8))


def train(args):
    if not args.cache.exists(): make_cache(args.cache,args.data)
    with np.load(args.cache) as c:
        xx=c["x"]; yy=c["y"]; cubes=c["cube"]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    torch.set_num_threads(4)
    net=FiberLikelihood().to(device)
    optim=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,args.steps,eta_min=args.lr*.08)
    train_ids=np.where(cubes!=args.valid_cube)[0]
    val_ids=np.where(cubes==args.valid_cube)[0]
    print("train",len(train_ids),"validation",len(val_ids),"device",device,flush=True)
    width=xx.shape[-1]
    xgrid=torch.arange(width,device=device)[None,None,None,:]
    ygrid=torch.arange(width,device=device)[None,None,:,None]
    def prepare(idx,aug):
        x=torch.from_numpy(xx[idx]).to(device,dtype=torch.float32)[:,None]/127.5-1
        xy=torch.from_numpy(yy[idx]).to(device)
        mask=((xy[:,:,0]>=0)&(xy[:,:,0]<width)&(xy[:,:,1]>=0)&(xy[:,:,1]<width))
        mask[:,[0,-1]]=False
        dist=(xgrid-xy[:,:,0,None,None])**2+(ygrid-xy[:,:,1,None,None])**2
        target=torch.exp(-dist/(2*1.65**2))
        target=target/target.sum((-1,-2),keepdim=True).clamp(min=1e-8)
        if aug:
            turns=int(rng.integers(4))
            x=torch.rot90(x,turns,(-2,-1))
            target=torch.rot90(target,turns,(-2,-1))
            if rng.random()<.5:
                x=x.flip(-1); target=target.flip(-1)
            if rng.random()<.5:
                x=x.flip(2); target=target.flip(1); mask=mask.flip(1)
            gain=torch.rand((len(idx),1,1,1,1),device=device)*.5+.75
            bias=torch.rand((len(idx),1,1,1,1),device=device)*.25-.125
            x=x*gain+bias+torch.randn_like(x)*float(rng.uniform(0,.05))
        return x,target,mask
    def loss(x,t,m):
        # One categorical distribution per cross-section, with finite-width manual labels.
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=net(x)
        logp=F.log_softmax(out.float().flatten(2),dim=-1)
        ce=-(t.flatten(2)*logp).sum(-1)
        return ce[m].mean()
    start=time.perf_counter()
    log=[]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for step in range(1,args.steps+1):
        idx=rng.choice(train_ids,args.batch,replace=len(train_ids)<args.batch)
        net.train(); x,t,m=prepare(idx,True)
        optim.zero_grad(set_to_none=True)
        l=loss(x,t,m)
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),3.0)
        optim.step(); scheduler.step()
        if step==1 or step%100==0 or step==args.steps:
            net.eval()
            with torch.inference_mode():
                fixed=val_ids[::max(1,len(val_ids)//40)][:40]
                vals=[float(loss(*prepare(fixed[i:i+args.batch],False))) for i in range(0,len(fixed),args.batch)]
            entry=dict(step=step,seconds=round(time.perf_counter()-start,1),loss=round(float(l.detach()),4),
                       validation=round(float(np.mean(vals)),4) if vals else None)
            log.append(entry);print(entry,flush=True)
            torch.save({"state":{k:v.detach().cpu() for k,v in net.state_dict().items()},"seed":args.seed,
                        "steps":step,"valid_cube":args.valid_cube},args.out)
            args.out.with_suffix(".log.json").write_text(json.dumps(log,indent=2))
    print("saved",args.out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path("data"))
    parser.add_argument("--cache",type=Path,default=Path("experiments/cache_s1.npz"))
    parser.add_argument("--out",type=Path,default=Path("weights/fiber_s1.pt"))
    parser.add_argument("--steps",type=int,default=600)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--lr",type=float,default=.0015)
    parser.add_argument("--seed",type=int,default=418)
    parser.add_argument("--valid-cube",type=int,default=1,help="Held-out Scroll 1 cube index, sorted by filename; -1 uses all Scroll 1 cubes.")
    train(parser.parse_args())
