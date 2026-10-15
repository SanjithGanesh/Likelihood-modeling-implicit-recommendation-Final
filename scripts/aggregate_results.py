from __future__ import annotations

import argparse, hashlib, json, math, shutil
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'step60-aggregation-v1'

CONTRACT_SHA = '651ac8f6b78073796858cb369cb99b9b16f7f20ef53125374e3405d39316e1a7'
FINAL_HP_SHA = '9aae84dd59d7c34c82a67563139bdb27256874ba79c645c749cbcf788d72c981'
FINAL_EVAL_SHA = '1874aeba214b3a049b439fa4d6b8aabbb8c20e6aa98fdfbeb94c0bb6348002f8'
STABILITY_SHA = '1e27f0b23684b83cccb2dd625110c597579e0d6f42489d191f06b45264360664'
PREFERENCE_SHA = '0ca60905d5cd71e9a589481e81919d772db1903517f03603d9490ffbfd7708bc'
STEP59_SHA = 'eae84c543f0e99440844d047ee0fa4b5217a89264e50c9f264ecbc3fdb2fa2d2'

FILES_TO_VERIFY = {
    ROOT/'config'/'step60_statistics_v1.yaml': CONTRACT_SHA,
    ROOT/'config'/'final_hyperparameters_v1.yaml': FINAL_HP_SHA,
    ROOT/'config'/'final_evaluation_v1.yaml': FINAL_EVAL_SHA,
    ROOT/'config'/'efemb_gaussian_stability_v1.yaml': STABILITY_SHA,
    ROOT/'config'/'preference_sensitivity_v1.yaml': PREFERENCE_SHA,
    ROOT/'docs'/'step59_temporal_decision_v1.md': STEP59_SHA,
}

PRIMARY_ID = 'final-ml1m-v1-1874aeba'
STABILITY_ID = 'stability-ml1m-v1-1e27f0b2'
PREFERENCE_ID = 'preference-ml1m-v1-0ca60905'
PRIMARY = ROOT/'results'/PRIMARY_ID
STABILITY = ROOT/'results'/STABILITY_ID
PREFERENCE = ROOT/'results'/PREFERENCE_ID

MODELS = (
    'Popularity','Random','PCA/SVD','BernoulliMF','PoissonMF','GaussianMF',
    'LogisticPCA','HPF','b-emb','p-emb','ap-emb','g-emb','ng-emb'
)
SEEDS = (2026, 2027, 2028)
KS = (5,10,20)
BASE = ('precision','recall','hit_rate','ndcg','mrr','map')
METRICS = tuple(f'{m}_at_{k}' for k in KS for m in BASE)
EXPECTED_USERS = 5851


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(',',':'), ensure_ascii=False)


def verify_hashes() -> None:
    for path, expected in FILES_TO_VERIFY.items():
        require(path.is_file(), f'Missing frozen file: {path}')
        actual = sha(path)
        require(actual == expected, f'SHA mismatch: {path}\nexpected={expected}\nactual={actual}')


def model_idx(model: str) -> int:
    return MODELS.index(model)


def load_run(run_root: Path, run_id: str, count: int, variant: str|None, target: str):
    exp = run_root/'experiments'
    dirs = sorted(p for p in exp.iterdir() if p.is_dir() and not p.name.startswith('.'))
    require(len(dirs)==count, f'{run_id}: expected {count} experiment dirs, got {len(dirs)}')
    out=[]
    for d in dirs:
        payload=json.loads((d/'result.json').read_text(encoding='utf-8'))
        require(payload.get('run_id')==run_id, f'{d.name}: run_id mismatch')
        require(payload.get('experiment_id')==d.name, f'{d.name}: experiment_id mismatch')
        r=payload['result']
        require(r['status']=='completed', f'{d.name}: not completed')
        require(r['target_split']==target, f'{d.name}: target mismatch')
        if variant is not None:
            require(r['variant']==variant, f'{d.name}: variant mismatch')
        rm=r.get('ranking_metrics',{})
        require(set(rm)==set(METRICS), f'{d.name}: ranking metric schema mismatch')
        require(all(math.isfinite(float(rm[m])) for m in METRICS), f'{d.name}: non-finite ranking metric')
        nm=r.get('native_metrics',{})
        require(all(math.isfinite(float(v)) for v in nm.values()), f'{d.name}: non-finite native metric')
        out.append({
            'experiment_id':d.name,'result_path':d/'result.json','model':r['model_name'],
            'family':r['family'],'variant':r['variant'],'seed':int(r['seed']),
            'target_split':r['target_split'],'ranking_metrics':{k:float(v) for k,v in rm.items()},
            'native_metrics':{k:float(v) for k,v in nm.items()},'metadata':dict(r.get('metadata',{}))
        })
    return sorted(out,key=lambda x:(model_idx(x['model']) if x['model'] in MODELS else 99,x['seed'],x['variant']))


def check_matrix(rows, label):
    obs={(r['model'],r['seed']) for r in rows}
    exp={(m,s) for m in MODELS for s in SEEDS}
    require(obs==exp, f'{label}: model/seed matrix mismatch; missing={sorted(exp-obs)} unexpected={sorted(obs-exp)}')


def validate_sources():
    primary=load_run(PRIMARY,PRIMARY_ID,39,'primary','test')
    pref=load_run(PREFERENCE,PREFERENCE_ID,39,'preference-r4','test')
    stab=load_run(STABILITY,STABILITY_ID,18,None,'validation')
    check_matrix(primary,'primary'); check_matrix(pref,'preference')
    cells={(r['model'],r['variant'],r['seed']) for r in stab}
    exp={(m,f'stability-{n}',s) for m in ('g-emb','ng-emb') for n in ('raw','sqrt','mean') for s in SEEDS}
    require(cells==exp,'stability matrix mismatch')
    pmap={(r['model'],r['seed']):r for r in primary}; qmap={(r['model'],r['seed']):r for r in pref}
    for key in pmap:
        p,q=pmap[key],qmap[key]
        require(p['family']==q['family'],f'{key}: family mismatch')
        require(canon(p['metadata'].get('hyperparameters',{}))==canon(q['metadata'].get('hyperparameters',{})),f'{key}: hyperparameters mismatch')
        require(int(p['metadata'].get('final_fit_rows',-1))==902826==int(q['metadata'].get('final_fit_rows',-2)),f'{key}: final fit rows mismatch')
    side=list((PREFERENCE/'preference_per_user').glob('*.csv'))
    diag=list((STABILITY/'stability_diagnostics').glob('*.json'))
    require(len(side)==39,f'preference sidecars: expected 39 got {len(side)}')
    require(len(diag)==18,f'stability diagnostics: expected 18 got {len(diag)}')
    return primary,pref,stab


def seed_frame(rows, stability=False):
    rec=[]
    for r in rows:
        x={'model':r['model'],'family':r['family'],'variant':r['variant'],'seed':r['seed'],'target_split':r['target_split'],'experiment_id':r['experiment_id']}
        if stability:
            require(r['variant'].startswith('stability-'),f"{r['experiment_id']}: bad stability variant")
            x['normalization']=r['variant'][len('stability-'):]
        x.update(r['ranking_metrics'])
        if stability:
            require({'rmse','mae'}<=set(r['native_metrics']),f"{r['experiment_id']}: missing rmse/mae")
            x['rmse']=r['native_metrics']['rmse']; x['mae']=r['native_metrics']['mae']
        rec.append(x)
    return pd.DataFrame(rec)


def summarize(frame: pd.DataFrame, groups, metrics):
    rows=[]
    grouped=frame.groupby(groups[0] if len(groups)==1 else list(groups),sort=False,dropna=False)
    for key,g in grouped:
        vals=(key,) if len(groups)==1 else tuple(key)
        row={k:v for k,v in zip(groups,vals,strict=True)}; row['n_seeds']=len(g)
        for m in metrics:
            s=g[m].astype(float)
            row[f'{m}_mean']=float(s.mean()); row[f'{m}_sd']=float(s.std(ddof=1)); row[f'{m}_min']=float(s.min()); row[f'{m}_max']=float(s.max())
        rows.append(row)
    return pd.DataFrame(rows)


def add_rank(frame: pd.DataFrame, mean_col: str, rank_col: str):
    order=sorted(((str(r['model']),float(r[mean_col])) for _,r in frame.iterrows()),key=lambda x:(-x[1],x[0]))
    ranks={m:i for i,(m,_) in enumerate(order,1)}
    out=frame.copy(); out[rank_col]=out['model'].map(ranks).astype(int); return out


def native_summary(rows):
    raw=[]
    for r in rows:
        for metric,val in sorted(r['native_metrics'].items()):
            raw.append({'model':r['model'],'family':r['family'],'seed':r['seed'],'metric':metric,'value':val})
    if not raw:
        return pd.DataFrame(columns=['model','family','metric','n_seeds','mean','sd','min','max'])
    df=pd.DataFrame(raw); out=[]
    for (m,f,metric),g in df.groupby(['model','family','metric'],sort=False):
        s=g['value'].astype(float); require(len(s)==3,f'{m}/{metric}: expected 3 seeds')
        out.append({'model':m,'family':f,'metric':metric,'n_seeds':3,'mean':float(s.mean()),'sd':float(s.std(ddof=1)),'min':float(s.min()),'max':float(s.max())})
    z=pd.DataFrame(out); z['_o']=z['model'].map(model_idx); return z.sort_values(['_o','metric']).drop(columns='_o').reset_index(drop=True)


def preference_user_summary(pref_rows):
    side=PREFERENCE/'preference_per_user'; by={m:[] for m in MODELS}
    for r in pref_rows: by[r['model']].append(r)
    ref=None; outputs=[]
    for model in MODELS:
        records=sorted(by[model],key=lambda r:r['seed']); require([r['seed'] for r in records]==list(SEEDS),f'{model}: bad seed inventory')
        arrays={m:[] for m in METRICS}; local=None
        for r in records:
            f=pd.read_csv(side/f"{r['experiment_id']}.csv").sort_values('userId').reset_index(drop=True)
            require(len(f)==EXPECTED_USERS,f"{r['experiment_id']}: expected {EXPECTED_USERS} users")
            meta=f[['userId','n_relevant','n_candidates']].astype(int)
            require((meta['n_relevant']>0).all(),f"{r['experiment_id']}: empty relevance")
            if local is None: local=meta
            else: require(meta.equals(local),f"{r['experiment_id']}: seed user metadata mismatch")
            if ref is None: ref=meta
            else: require(meta.equals(ref),f"{r['experiment_id']}: model user metadata mismatch")
            for m in METRICS:
                a=f[m].to_numpy(dtype=float); require(np.isfinite(a).all(),f"{r['experiment_id']}: non-finite {m}"); arrays[m].append(a)
        out=local.copy(); out.insert(0,'model',model); out['n_seeds']=3
        for m in METRICS:
            a=np.stack(arrays[m]); out[f'{m}_mean']=a.mean(0); out[f'{m}_sd']=a.std(0,ddof=1)
        outputs.append(out)
    result=pd.concat(outputs,ignore_index=True); require(len(result)==len(MODELS)*EXPECTED_USERS,'user-model summary row mismatch'); return result


def flatten(x: Mapping, prefix=''):
    out={}
    for k in sorted(x):
        name=f'{prefix}.{k}' if prefix else str(k); v=x[k]
        if isinstance(v,Mapping): out.update(flatten(v,name))
        elif isinstance(v,(list,tuple)): out[name]=canon(v)
        else: out[name]=v
    return out


def stability_diagnostics(stab_rows):
    paths=sorted((STABILITY/'stability_diagnostics').glob('*.json')); ids={r['experiment_id'] for r in stab_rows}; rows=[]
    for p in paths:
        x=json.loads(p.read_text(encoding='utf-8')); require(x.get('experiment_id') in ids,f'{p.name}: id mismatch'); require(x.get('test_metrics_computed') is False,f'{p.name}: test metrics flag not false'); rows.append(flatten(x))
    return pd.DataFrame(rows)


def build_outputs(primary,pref,stab):
    ps=seed_frame(primary); qs=seed_frame(pref); ss=seed_frame(stab,stability=True)
    p=sumry=summarize(ps,('model',),METRICS); p=add_rank(p,'ndcg_at_10_mean','primary_rank')
    q=summarize(qs,('model',),METRICS); q=add_rank(q,'ndcg_at_10_mean','preference_rank')
    for f in (ps,qs,p,q):
        f['_o']=f['model'].map(model_idx); f.sort_values(['_o']+(['seed'] if 'seed' in f.columns else []),inplace=True,kind='stable'); f.drop(columns='_o',inplace=True); f.reset_index(drop=True,inplace=True)
    cmp=p[['model','ndcg_at_10_mean','ndcg_at_10_sd','primary_rank']].rename(columns={'ndcg_at_10_mean':'primary_ndcg_at_10_mean','ndcg_at_10_sd':'primary_ndcg_at_10_sd'}).merge(
        q[['model','ndcg_at_10_mean','ndcg_at_10_sd','preference_rank']].rename(columns={'ndcg_at_10_mean':'preference_ndcg_at_10_mean','ndcg_at_10_sd':'preference_ndcg_at_10_sd'}),on='model',validate='one_to_one')
    cmp['absolute_difference']=cmp['preference_ndcg_at_10_mean']-cmp['primary_ndcg_at_10_mean']
    ranks=cmp[['model','primary_rank','preference_rank','primary_ndcg_at_10_mean','preference_ndcg_at_10_mean']].copy(); ranks['rank_shift']=ranks['preference_rank']-ranks['primary_rank']
    st=summarize(ss,('model','normalization'),('ndcg_at_10','rmse','mae'))
    norm_order={'raw':0,'sqrt':1,'mean':2}
    for f in (ss,st):
        f['_m']=f['model'].map({'g-emb':0,'ng-emb':1}); f['_n']=f['normalization'].map(norm_order); cols=['_m','_n']+(['seed'] if 'seed' in f.columns else []); f.sort_values(cols,inplace=True,kind='stable'); f.drop(columns=['_m','_n'],inplace=True); f.reset_index(drop=True,inplace=True)
    d=stability_diagnostics(stab)
    if {'model_name','context_normalization','seed'}<=set(d.columns): d.sort_values(['model_name','context_normalization','seed'],inplace=True,kind='stable')
    return {
        'primary_seed_metrics.csv':ps,'primary_summary.csv':p,'primary_native_summary.csv':native_summary(primary),
        'preference_seed_metrics.csv':qs,'preference_summary.csv':q,'preference_user_model_summary.csv':preference_user_summary(pref),
        'primary_vs_preference_ndcg10.csv':cmp,'model_rank_summary.csv':ranks,
        'stability_seed_metrics.csv':ss,'stability_summary.csv':st,'stability_diagnostics_flat.csv':d,
    }


def combined_digest(root: Path, paths):
    h=hashlib.sha256()
    for p in sorted(Path(x).resolve() for x in paths):
        rel=p.relative_to(root.resolve()).as_posix(); h.update(rel.encode()); h.update(b'\0'); h.update(sha(p).encode()); h.update(b'\n')
    return h.hexdigest()


def write_csv(df,path): df.to_csv(path,index=False,encoding='utf-8',float_format='%.17g')


def execute(primary,pref,stab):
    require(not OUT.exists(),f'Output directory already exists: {OUT}')
    tmp=OUT.parent/f'.{OUT.name}.tmp'; require(not tmp.exists(),f'Staging directory already exists: {tmp}')
    outputs=build_outputs(primary,pref,stab)
    digests={
        'primary_inputs_sha256':combined_digest(PRIMARY,[r['result_path'] for r in primary]),
        'preference_inputs_sha256':combined_digest(PREFERENCE,[r['result_path'] for r in pref]+list((PREFERENCE/'preference_per_user').glob('*.csv'))),
        'stability_inputs_sha256':combined_digest(STABILITY,[r['result_path'] for r in stab]+list((STABILITY/'stability_diagnostics').glob('*.json'))),
    }
    try:
        tmp.mkdir()
        artifacts={}
        for name,df in outputs.items():
            p=tmp/name; write_csv(df,p); artifacts[name]={'rows':len(df),'columns':len(df.columns),'sha256':sha(p)}
        manifest={
            'schema_version':'1.0','freeze_id':'paper2-step60-aggregation-v1','statistics_contract_sha256':CONTRACT_SHA,
            'aggregation_script_sha256':sha(Path(__file__).resolve()),'source_runs':{'primary':PRIMARY_ID,'stability':STABILITY_ID,'preference':PREFERENCE_ID},
            'source_protocol_hashes':{'final_hyperparameters_sha256':FINAL_HP_SHA,'final_evaluation_sha256':FINAL_EVAL_SHA,'stability_protocol_sha256':STABILITY_SHA,'preference_protocol_sha256':PREFERENCE_SHA,'step59_decision_sha256':STEP59_SHA},
            'source_input_digests':digests,'inventory':{'primary_experiments':39,'preference_experiments':39,'stability_experiments':18,'preference_users_per_sidecar':EXPECTED_USERS},
            'statistical_definitions':{'seed_summary':'mean, sample standard deviation (ddof=1), minimum, maximum across frozen seeds','primary_reporting_metric':'ndcg_at_10','rank_tie_break':'model_name_ascending','preference_difference':'preference_mean_minus_primary_mean','rank_shift':'preference_rank_minus_primary_rank','inferential_tests':False,'confidence_intervals':False,'p_values':False},
            'artifacts':artifacts,
        }
        (tmp/'aggregation_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8'); tmp.replace(OUT)
    except Exception:
        if tmp.exists(): shutil.rmtree(tmp)
        raise
    return OUT


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
    verify_hashes(); primary,pref,stab=validate_sources()
    print('='*92); print('STEP 60C AGGREGATION PREFLIGHT: PASS'); print('='*92)
    print('statistics_contract_sha256:',CONTRACT_SHA); print('primary experiments:',len(primary)); print('preference experiments:',len(pref)); print('stability experiments:',len(stab)); print('primary/preference hyperparameter identities: 39 / 39 matched'); print('primary inferential tests: DISABLED'); print('preference per-user role: descriptive distribution only'); print(); print('NO MODEL WAS FIT.'); print('NO SCORE WAS RECOMPUTED.'); print('NO SOURCE RESULT WAS MODIFIED.'); print('='*92)
    if not args.execute: return 0
    out=execute(primary,pref,stab); p=pd.read_csv(out/'primary_summary.csv'); q=pd.read_csv(out/'preference_summary.csv')
    print(); print('='*92); print('STEP 60C AGGREGATION EXECUTION COMPLETE'); print('='*92); print('Output directory:',out)
    print('\nPRIMARY NDCG@10 MEAN ± SAMPLE SD')
    for _,r in p.sort_values('primary_rank').iterrows(): print(f"{int(r.primary_rank):2d}. {r['model']:<16} {r.ndcg_at_10_mean:.6f} ± {r.ndcg_at_10_sd:.6f}")
    print('\nPREFERENCE NDCG@10 MEAN ± SAMPLE SD')
    for _,r in q.sort_values('preference_rank').iterrows(): print(f"{int(r.preference_rank):2d}. {r['model']:<16} {r.ndcg_at_10_mean:.6f} ± {r.ndcg_at_10_sd:.6f}")
    print(); print('='*92); print('STEP 60C COMPLETE'); print('='*92); return 0

if __name__=='__main__': raise SystemExit(main())
