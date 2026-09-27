"""Portable, CPU-only evaluation. Requires Python 3.10+ and numpy. No project imports."""
import argparse, csv, json
from pathlib import Path
import numpy as np

def rows(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def unit(x):
    x=np.asarray(x,np.float64);n=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or (n==0).any():raise ValueError('Invalid/zero embedding')
    return x/n
def metrics(cases):
    classes=sorted({r['class_id'] for r in cases});out={'query_count':len(cases)}
    for k in [1,5,10,20]:
        hits=[int(r['true_rank'])<=k for r in cases];out[f'hits_at_{k}']=sum(hits);out[f'recall_at_{k}']=sum(hits)/len(hits)
        out[f'macro_recall_at_{k}']=float(np.mean([np.mean([h for r,h in zip(cases,hits) if r['class_id']==c]) for c in classes]))
    return out
def rank(split,features,metric='cosine',bank=None,mode=None):
    with np.load(features,allow_pickle=False) as z:
        lookup={k:i for i,k in enumerate(z['ids'].tolist())}
        if len(lookup)!=len(z['ids']):raise ValueError('Duplicate feature IDs')
        f=np.asarray(z['features'][[lookup[r['id']] for r in split]],np.float64)
    gi=[i for i,r in enumerate(split) if r['role']=='gallery'];qi=[i for i,r in enumerate(split) if r['role']=='query'];g=[split[i] for i in gi];q=[split[i] for i in qi]
    if metric=='cosine':f=unit(f);scores=f[qi]@f[gi].T
    else:
        scores=np.empty((len(q),len(g)))
        for i,j in enumerate(qi):scores[i]=-np.square(f[gi]-f[j]).sum(1)
    if bank:
        if metric!='cosine':raise ValueError('B bank is frozen SigLIP cosine only')
        with np.load(bank,allow_pickle=False) as z:
            ix={(s,int(v)):i for i,(s,v) in enumerate(zip(z['source_ids'],z['view_indices']))}
            v=unit(z['features'][[ix[r['id'],j] for r in g for j in range(6)]].reshape(len(g),6,-1))
        if mode=='mean':scores=f[qi]@unit(np.concatenate([f[gi,None,:],v],axis=1).mean(1)).T
        elif mode in ['max','views-only']:
            synth=np.maximum.reduce([f[qi]@v[:,j].T for j in range(6)]);scores=np.maximum(scores,synth) if mode=='max' else synth
        else:raise ValueError('Specify max, mean, or views-only')
    truth={r['class_id']:i for i,r in enumerate(g)};cases=[]
    assert len(truth)==len(g)
    for item,s in zip(q,scores):
        order=np.lexsort((np.arange(len(g)),-s));rank=int(np.flatnonzero(order==truth[item['class_id']])[0])+1
        cases.append(dict(query_id=item['id'],class_id=item['class_id'],true_gallery_id=g[truth[item['class_id']]]['id'],true_rank=rank,top20_gallery_ids=[g[j]['id'] for j in order[:20]]))
    return dict(metrics=metrics(cases),cases=cases)
def paired(targets,reference,draws=5000,seed=42):
    base={r['query_id']:r for r in reference};ids=sorted(base);classes=np.array([base[k]['class_id'] for k in ids]);b=np.array([int(base[k]['true_rank'])==1 for k in ids]);diff=[];per=[]
    for run in targets:
        d={r['query_id']:r for r in run};assert set(d)==set(base)
        assert all(d[k]['class_id']==base[k]['class_id'] for k in ids)
        a=np.array([int(d[k]['true_rank'])==1 for k in ids]);diff.append(a.astype(float)-b)
        per.append(dict(rescued=int((a&~b).sum()),harmed=int((~a&b).sum()),target_correct=int(a.sum()),reference_correct=int(b.sum())))
    delta=np.mean(diff,axis=0);unique=np.unique(classes);counts=np.array([(classes==c).sum() for c in unique]);sums=np.array([delta[classes==c].sum() for c in unique]);means=sums/counts
    rng=np.random.default_rng(seed);micro=[];macro=[]
    for start in range(0,draws,250):
        ix=rng.integers(0,len(unique),(min(250,draws-start),len(unique)));micro.extend(sums[ix].sum(1)/counts[ix].sum(1));macro.extend(means[ix].mean(1))
    return dict(per_run=per,query_weighted=dict(delta_pp=float(100*sums.sum()/counts.sum()),ci95_pp=(100*np.percentile(micro,[2.5,97.5])).tolist()),class_macro=dict(delta_pp=float(100*means.mean()),ci95_pp=(100*np.percentile(macro,[2.5,97.5])).tolist()),draws=draws,seed=seed,unit='class cluster',training_seed_uncertainty_included=False,multiplicity_corrected=False)
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('rank');r.add_argument('--split',required=True);r.add_argument('--features',required=True);r.add_argument('--metric',choices=['cosine','sqeuclidean'],default='cosine');r.add_argument('--bank');r.add_argument('--mode',choices=['max','mean','views-only']);r.add_argument('--compare-csv');r.add_argument('--output',required=True)
    s=sub.add_parser('paired');s.add_argument('--targets',nargs='+',required=True);s.add_argument('--reference',required=True);s.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='rank':
        result=rank(rows(a.split),a.features,a.metric,a.bank,a.mode)
        if a.compare_csv:
            expected={r['query_id']:r for r in rows(a.compare_csv)};assert len(expected)==len(result['cases'])
            for r in result['cases']:
                ex=expected[r['query_id']];assert r['true_rank']==int(ex['true_rank']),r['query_id'];assert r['top20_gallery_ids']==[ex[f'top{i:02d}_gallery_id'] for i in range(1,21)],r['query_id']
            result['csv_exact_query_ranks_and_top20']=len(expected)
    else:result=paired([rows(p) for p in a.targets],rows(a.reference))
    with open(a.output,'x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result.get('metrics',result),ensure_ascii=True))
if __name__=='__main__':main()
