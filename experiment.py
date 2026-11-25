"""Reproduce one anchor-only tracing comparison per labeled fiber, grouped by physical cube."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
from pathlib import Path

import numpy as np

from dataset import tasks_for_cube
from fiber_trace import estimate, read_volume


def errors(reference: np.ndarray, path: np.ndarray) -> np.ndarray:
    # Only original manual interior nodes. All distance is three-dimensional
    # and measured against continuous line segments, not a set of sampled knots.
    hidden=reference[1:-1]
    starts=path[:-1]
    delta=path[1:]-starts
    offset=hidden[:,None,:]-starts[None,:,:]
    fraction=np.clip(np.sum(offset*delta,axis=-1)/np.maximum(np.sum(delta*delta,axis=-1),1e-9),0,1)
    return np.linalg.norm(offset-fraction[...,None]*delta,axis=-1).min(axis=1)


def summarize(rows):
    methods=sorted(set(r["method"] for r in rows))
    results={}
    for m in methods:
        group=[r for r in rows if r["method"]==m]
        err=np.array([r["mean_vox"] for r in group])
        tails=np.array([r["p90_vox"] for r in group])
        results[m]={"n":len(group),"mean_vox":float(err.mean()),"median_vox":float(np.median(err)),
                    "p90_of_means":float(np.percentile(err,90)),
                    "within_3":float(np.mean(tails<=3)), "bad_over_5":float(np.mean(tails>5))}
    return results


def run(args):
    start_time=time.perf_counter()
    rows=[]
    dumps=[]
    data=Path(args.data)
    predictor=None
    if args.model:
        from fiber_model import Predictor
        predictor=Predictor(args.model)
    args.output.mkdir(parents=True,exist_ok=True)
    tiffs=sorted(data.glob(args.split+"_*.tif"))
    if args.cube:
        tiffs=[p for p in tiffs if args.cube in p.stem]
    if not tiffs:
        raise ValueError("Fetch the official CT images first")
    for tiff in tiffs:
        volume=read_volume(tiff)
        tasks=tasks_for_cube(tiff,args.limit)
        def one(task):
            before=time.perf_counter()
            paths,diagnostics=estimate(volume,task.path[0],task.path[-1],strengths=args.strengths,
                                       predictor=predictor,neural_strengths=args.neural_strengths,refit=args.refit)
            elapsed=time.perf_counter()-before
            output=[]
            for name,path in paths.items():
                err=errors(task.path,path)
                back_err=errors(path,task.path)
                output.append({"cube":task.cube,"tree":task.tree,"key":task.key,"method":name,
                               "nodes":len(err),"mean_vox":float(err.mean()),"p90_vox":float(np.percentile(err,90)),
                               "max_vox":float(err.max()),"reverse_mean_vox":float(back_err.mean()),"seconds_all_methods":elapsed,**diagnostics})
            dump={"cube":task.cube,"tree":task.tree,"key":task.key,"annotation":task.path.round(3).tolist(),
                  "paths":{n:v.astype(float).round(3).tolist() for n,v in paths.items()},"diagnostics":diagnostics}
            return output,dump
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            cube_rows=[]
            for result,dump in pool.map(one,tasks):
                cube_rows.extend(result)
                dumps.append(dump)
            rows.extend(cube_rows)
        by_method=summarize(cube_rows)
        print(tiff.stem,len(tasks),{k:round(v["mean_vox"],3) for k,v in by_method.items()},flush=True)
    with (args.output/"per_trace.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output/"paths.json").write_text(json.dumps(dumps,separators=(",",":")),encoding="utf-8")
    summary={"split":args.split,"limit":args.limit,"strengths":args.strengths,
             "model":str(args.model) if args.model else None,"neural_strengths":args.neural_strengths,
             "seconds":time.perf_counter()-start_time,"results":summarize(rows)}
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",default="data")
    parser.add_argument("--split",choices=["s1","s5"],required=True)
    parser.add_argument("--cube",help="Optional cube substring for development-only comparisons.")
    parser.add_argument("--limit",type=int,default=0,help="Fixed deterministic cap per cube; 0 uses every eligible tree.")
    parser.add_argument("--strengths",type=float,nargs="+",default=[3])
    parser.add_argument("--neural-strengths",type=float,nargs="+",default=[1])
    parser.add_argument("--model",type=Path)
    parser.add_argument("--refit",action="store_true")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
