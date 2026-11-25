"""Connect two clicked points on a papyrus fiber in a real 3-D CT volume.

Input volumes have array order z,y,x. Clicks and output paths use x,y,z in voxels.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates
from registration import registrations, solve_bridge

cv2.setNumThreads(1)


def read_volume(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        if img.mode not in ("L", "I;16", "I;16L", "I;16B", "I", "F"):
            raise ValueError("Expected a single-channel CT stack")
        array = np.empty((img.n_frames, img.height, img.width), dtype=np.asarray(img).dtype)
        for z in range(len(array)):
            img.seek(z)
            array[z] = np.asarray(img)
    return array


@dataclass
class Tube:
    images: np.ndarray
    centers: np.ndarray
    u: np.ndarray
    v: np.ndarray
    radius: int

    def xyz(self, xy: np.ndarray) -> np.ndarray:
        return self.centers + (xy[:, :1] - self.radius) * self.u + (xy[:, 1:] - self.radius) * self.v


def make_tube(volume: np.ndarray, a, b, radius: int = 18, spacing: float = 3.0, planes: int | None = None) -> Tube:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    shape = np.array(volume.shape[::-1])
    if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a,b]).all():
        raise ValueError("Use two finite x,y,z triples")
    if np.any(np.minimum(a, b) < 0) or np.any(np.maximum(a, b) > shape - 1):
        raise ValueError("The two clicks must be inside the volume")
    length = np.linalg.norm(b - a)
    if length < 2 or radius < 5 or spacing <= 0:
        raise ValueError("The two clicks should be at least two voxels apart")
    tangent = (b-a) / length
    axis = np.eye(3, dtype=np.float32)[np.argmin(np.abs(tangent))]
    u = np.cross(tangent, axis)
    u /= np.linalg.norm(u)
    v = np.cross(tangent, u)
    count = planes if planes is not None else max(3, int(np.ceil(length / spacing)) + 1)
    centers = np.linspace(a, b, count, dtype=np.float32)
    return sample_planes(volume,centers,u,v,radius)


def sample_planes(volume,centers,u,v,radius):
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1].astype(np.float32)
    uu=np.asarray(u).reshape(-1,1,1,3)
    vv=np.asarray(v).reshape(-1,1,1,3)
    xyz = centers[:, None, None, :] + xx[None,...,None]*uu + yy[None,...,None]*vv
    coords = np.moveaxis(xyz[..., ::-1], -1, 0)
    raw = map_coordinates(volume, coords, output=np.float32, order=1, mode="nearest", prefilter=False)
    lo, hi = np.percentile(raw, (1, 99))
    scaled = np.clip((raw-lo)*(255/max(hi-lo, 1e-6)), 0, 255).astype(np.uint8)
    return Tube(scaled, centers, u, v, radius)


def curved_planes(volume,tube,path):
    centers=gaussian_filter1d(path,1.7,axis=0,mode="nearest")
    centers[[0,-1]]=path[[0,-1]]
    tangents=np.gradient(centers,axis=0)
    tangents/=np.linalg.norm(tangents,axis=1,keepdims=True).clip(min=1e-7)
    u=np.zeros_like(centers)
    previous=tube.u
    for k,tan in enumerate(tangents):
        transverse=previous-(previous@tan)*tan
        u[k]=transverse/max(np.linalg.norm(transverse),1e-7)
        previous=u[k]
    return sample_planes(volume,centers,u,np.cross(tangents,u),tube.radius)


def flow_maps(images: np.ndarray):
    def flow(lhs, rhs):
        return cv2.calcOpticalFlowFarneback(lhs, rhs, None, .5, 2, 13, 3, 5, 1.1, 0)
    forward = np.stack([flow(lhs, rhs) for lhs, rhs in zip(images[:-1], images[1:])])
    reverse = np.stack([flow(rhs, lhs) for lhs, rhs in zip(images[:-1], images[1:])])
    return forward, reverse


def follow(field: np.ndarray, start: float) -> np.ndarray:
    out = np.empty((len(field)+1, 2), dtype=np.float32)
    out[0] = start
    for k, displacement in enumerate(field):
        x,y = np.clip(out[k], 0, np.array(displacement.shape[:2][::-1])-1.001)
        i,j = int(x),int(y)
        dx,dy = x-i,y-j
        step = ((1-dy)*((1-dx)*displacement[j,i]+dx*displacement[j,i+1]) +
                dy*((1-dx)*displacement[j+1,i]+dx*displacement[j+1,i+1]))
        out[k+1] = np.clip(out[k]+step, 0, np.array(displacement.shape[:2][::-1])-1)
    return out


def endpoint_similarity(images: np.ndarray) -> np.ndarray:
    """Match real cross-sectional texture around each click, without assuming bright ink/fibers."""
    z, h, w = images.shape
    cy, cx = h//2, w//2
    weight = np.linspace(0,1,z,dtype=np.float32)[:,None,None]
    score = np.zeros((z,h,w),dtype=np.float32)
    smooth = np.stack([cv2.GaussianBlur(im,(3,3),.7) for im in images])
    for radius, scale in ((3,.55), (6,.45)):
        pieces = []
        for im in (smooth[0],smooth[-1]):
            template = im[cy-radius:cy+radius+1,cx-radius:cx+radius+1]
            if template.std() < 1:
                pieces.append(np.zeros((z,h,w), np.float32))
                continue
            stack = []
            for plane in smooth:
                pad = cv2.copyMakeBorder(plane,radius,radius,radius,radius,cv2.BORDER_REFLECT_101)
                stack.append(cv2.matchTemplate(pad, template, cv2.TM_CCOEFF_NORMED))
            pieces.append(np.stack(stack))
        score += scale * ((1-weight)*pieces[0] + weight*pieces[1])
    return score


def optimal_bridges(forward: np.ndarray, reverse: np.ndarray, similarity: np.ndarray, strengths) -> dict:
    """Minimum-cost endpoint-constrained paths on resliced CT cross-sections.

    Transitions penalize disagreement with *both* image-flow estimates. The
    node cost rewards appearance agreement with the clicked fiber ends.
    """
    planes,height,width = similarity.shape
    center = (height//2,width//2)
    strengths = np.asarray(strengths,dtype=np.float32)
    count=len(strengths)
    cost=np.full((count,height,width),np.inf,dtype=np.float32)
    cost[:,center[0],center[1]]=0
    directions=[(dy,dx) for dy in range(-2,3) for dx in range(-2,3)]
    back=np.zeros((planes,count,height,width),dtype=np.uint8)
    slices=[]
    for dy,dx in directions:
        old=(slice(max(0,-dy),min(height,height-dy)),slice(max(0,-dx),min(width,width-dx)))
        new=(slice(max(0,dy),min(height,height+dy)),slice(max(0,dx),min(width,width+dx)))
        slices.append((old,new,dy,dx))
    for k in range(1,planes):
        best=np.full_like(cost,np.inf)
        indices=back[k]
        f=forward[k-1]
        r=reverse[k-1]
        for idx,(old,new,dy,dx) in enumerate(slices):
            fx,fy=f[old][...,0],f[old][...,1]
            rx,ry=r[new][...,0],r[new][...,1]
            penalty=.5*((dx-fx)**2+(dy-fy)**2+(dx+rx)**2+(dy+ry)**2)+.08*(dx*dx+dy*dy)
            alt=cost[(slice(None),)+old]+penalty[None]
            dst=best[(slice(None),)+new]
            improve=alt<dst
            np.copyto(indices[(slice(None),)+new],idx,where=improve)
            np.minimum(dst,alt,out=dst)
        cost=best-strengths[:,None,None]*similarity[k]
    paths={}
    for m,power in enumerate(strengths):
        yy,xx=center
        path=np.empty((planes,2),np.float32)
        path[-1]=[xx,yy]
        for k in range(planes-1,0,-1):
            dy,dx=directions[back[k,m,yy,xx]]
            yy-=dy
            xx-=dx
            path[k-1]=[xx,yy]
        path=gaussian_filter1d(path,.85,axis=0,mode="nearest")
        path[[0,-1]]=center[::-1]
        paths[float(power)]=path
    return paths


def refine_local(path, relative_mass):
    """Subvoxel center of the posterior near the connected route, preserving both clicks."""
    _,h,w=relative_mass.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    d2=(xx[None]-path[:,0,None,None])**2+(yy[None]-path[:,1,None,None])**2
    mass=relative_mass*np.exp(-d2/(2*1.6**2))
    norm=mass.sum(axis=(1,2)).clip(min=1e-20)
    refined=np.column_stack([(mass*xx).sum((1,2))/norm,(mass*yy).sum((1,2))/norm])
    refined=gaussian_filter1d(refined,.7,axis=0,mode="nearest")
    refined[[0,-1]]=path[[0,-1]]
    return refined


def estimate(volume: np.ndarray, a, b, strengths=(1.0,), radius=18, predictor=None, neural_strengths=(1.0,), refit=False):
    tube=make_tube(volume,a,b,radius=radius)
    forward,reverse=flow_maps(tube.images)
    p=follow(forward,tube.radius)
    q=follow(reverse[::-1],tube.radius)[::-1]
    t=np.linspace(0,1,len(p),dtype=np.float32)[:,None]
    linear=np.full_like(p,tube.radius)
    classic=p+(tube.radius-p[-1])*t
    blend=(1-t)*p+t*q
    sim=endpoint_similarity(tube.images)
    paths={"line":linear,"forward_flow":classic,"two_way_flow":blend}
    for strength,path in optimal_bridges(forward,reverse,sim,strengths).items():
        paths[f"bridge_{strength:g}"]=path
    links=registrations(tube.images,forward,reverse)
    paths["multiscale_flow"]=solve_bridge(blend,links)
    paths["multiscale_texture"]=solve_bridge(paths["multiscale_flow"],links,3*sim,.20)
    diagnostics={"direction_disagreement_vox":float(np.mean(np.linalg.norm(p-q,axis=1))),
                 "end_drift_vox":float((np.linalg.norm(p[-1]-tube.radius)+np.linalg.norm(q[0]-tube.radius))/2)}
    if predictor is not None:
        logits=predictor.logits(tube.images)
        logits-=logits.max(axis=(1,2),keepdims=True)
        mass=np.exp(logits)
        prob=mass/mass.sum(axis=(1,2),keepdims=True)
        yy,xx=np.mgrid[:logits.shape[1],:logits.shape[2]].astype(np.float32)
        mean=np.column_stack([(prob*xx).sum((1,2)),(prob*yy).sum((1,2))])
        mean=gaussian_filter1d(mean,.75,axis=0)
        mean[[0,-1]]=tube.radius
        paths["learned_mean"]=mean
        paths["multiscale_learned"]=solve_bridge(paths["multiscale_flow"],links,logits,.35)
        diagnostics["posterior_spread_vox"]=float(np.mean(np.sqrt((prob*((xx[None]-mean[:,0,None,None])**2+(yy[None]-mean[:,1,None,None])**2)).sum((1,2)))[1:-1]))
        for power,route in optimal_bridges(forward,reverse,logits,neural_strengths).items():
            paths[f"learned_bridge_{power:g}"]=refine_local(route,mass)
    world={k:tube.xyz(v) for k,v in paths.items()}
    if refit:
        first=world["multiscale_learned" if predictor is not None else "multiscale_texture"]
        aligned=curved_planes(volume,tube,first)
        af,ar=flow_maps(aligned.images)
        ap=follow(af,aligned.radius)
        aq=follow(ar[::-1],aligned.radius)[::-1]
        alinks=registrations(aligned.images,af,ar)
        score=predictor.logits(aligned.images) if predictor is not None else 3*endpoint_similarity(aligned.images)
        adjusted=aligned.xyz(solve_bridge((1-t)*ap+t*aq,alinks,score,.35 if predictor is not None else .20))
        world["refit"]=adjusted
        world["refit_blend"]=(first+adjusted)*.5
        diagnostics["refit_change_vox"]=float(np.mean(np.linalg.norm(adjusted-first,axis=1)))
    return world,diagnostics


def write_nml(path: Path, xyz: np.ndarray, experiment: str, voxel_size_um: float):
    root=ET.Element("things")
    params=ET.SubElement(root,"parameters")
    ET.SubElement(params,"experiment",name=experiment)
    ET.SubElement(params,"scale",x=str(voxel_size_um),y=str(voxel_size_um),z=str(voxel_size_um),unit="micrometer")
    tree=ET.SubElement(root,"thing",id="1",name="Proposed fiber bridge")
    nodes=ET.SubElement(tree,"nodes")
    edges=ET.SubElement(tree,"edges")
    for i,point in enumerate(xyz,1):
        attrs={k:str(int(round(float(v)))) for k,v in zip("xyz",point)}
        ET.SubElement(nodes,"node",id=str(i),radius="1.0",**attrs)
        if i>1: ET.SubElement(edges,"edge",source=str(i-1),target=str(i))
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume",type=Path)
    parser.add_argument("--start",type=float,nargs=3,required=True,metavar=("X","Y","Z"))
    parser.add_argument("--end",type=float,nargs=3,required=True,metavar=("X","Y","Z"))
    parser.add_argument("--origin",type=float,nargs=3,default=(0,0,0),help="Global x y z of local voxel (0,0,0); added only for export.")
    parser.add_argument("--experiment",required=True,help="WEBKNOSSOS dataset name in the output NML")
    parser.add_argument("--voxel-size-um",type=float,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model",type=Path,help="Optional Scroll 1 weights for endpoint-conditioned fiber tracing.")
    args=parser.parse_args()
    predictor=None
    if args.model:
        from fiber_model import Predictor
        predictor=Predictor(args.model)
    paths,quality=estimate(read_volume(args.volume),args.start,args.end,predictor=predictor)
    method="learned_bridge_1" if predictor else "bridge_1"
    write_nml(args.out,paths[method]+args.origin,args.experiment,args.voxel_size_um)
    print(quality)
