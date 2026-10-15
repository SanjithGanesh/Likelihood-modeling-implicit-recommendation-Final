from __future__ import annotations
import pandas as pd
import pytest
from scripts import aggregate_results as agg


def test_contract_hash():
    assert agg.CONTRACT_SHA == '651ac8f6b78073796858cb369cb99b9b16f7f20ef53125374e3405d39316e1a7'


def test_metric_inventory():
    assert len(agg.METRICS)==18
    assert set(agg.METRICS)=={f'{m}_at_{k}' for k in (5,10,20) for m in ('precision','recall','hit_rate','ndcg','mrr','map')}


def test_summary_uses_sample_sd():
    f=pd.DataFrame({'model':['A']*3,'ndcg_at_10':[1.0,2.0,3.0]})
    s=agg.summarize(f,('model',),('ndcg_at_10',)).iloc[0]
    assert s.ndcg_at_10_mean==pytest.approx(2.0)
    assert s.ndcg_at_10_sd==pytest.approx(1.0)
    assert s.ndcg_at_10_min==pytest.approx(1.0)
    assert s.ndcg_at_10_max==pytest.approx(3.0)


def test_rank_tie_break_model_name_ascending():
    f=pd.DataFrame({'model':['Zulu','Alpha','Beta'],'ndcg_at_10_mean':[.5,.5,.4]})
    r=agg.add_rank(f,'ndcg_at_10_mean','rank'); d=dict(zip(r.model,r['rank']))
    assert d=={'Zulu':2,'Alpha':1,'Beta':3}


def test_matrix_check_rejects_missing():
    rows=[{'model':m,'seed':s} for m in agg.MODELS for s in agg.SEEDS]
    agg.check_matrix(rows,'x')
    with pytest.raises(RuntimeError,match='matrix mismatch'):
        agg.check_matrix(rows[:-1],'x')


def test_flatten_nested_diagnostic():
    assert agg.flatten({'b':{'z':2,'a':[3,1]},'a':True})=={'a':True,'b.a':'[3,1]','b.z':2}


def test_preference_difference_and_rank_shift():
    p=pd.DataFrame({'model':['A','B'],'ndcg_at_10_mean':[.5,.4],'ndcg_at_10_sd':[.01,.02],'primary_rank':[1,2]})
    q=pd.DataFrame({'model':['A','B'],'ndcg_at_10_mean':[.35,.45],'ndcg_at_10_sd':[.03,.04],'preference_rank':[2,1]})
    cmp=p[['model','ndcg_at_10_mean','ndcg_at_10_sd','primary_rank']].rename(columns={'ndcg_at_10_mean':'primary_ndcg_at_10_mean','ndcg_at_10_sd':'primary_ndcg_at_10_sd'}).merge(q[['model','ndcg_at_10_mean','ndcg_at_10_sd','preference_rank']].rename(columns={'ndcg_at_10_mean':'preference_ndcg_at_10_mean','ndcg_at_10_sd':'preference_ndcg_at_10_sd'}),on='model')
    cmp['absolute_difference']=cmp.preference_ndcg_at_10_mean-cmp.primary_ndcg_at_10_mean
    cmp['rank_shift']=cmp.preference_rank-cmp.primary_rank
    a=cmp.set_index('model').loc['A']; b=cmp.set_index('model').loc['B']
    assert a.absolute_difference==pytest.approx(-.15); assert b.absolute_difference==pytest.approx(.05)
    assert a.rank_shift==1; assert b.rank_shift==-1


def test_combined_digest_order_independent(tmp_path):
    a=tmp_path/'a.txt'; b=tmp_path/'b.txt'; a.write_text('x'); b.write_text('y')
    x=agg.combined_digest(tmp_path,[a,b]); y=agg.combined_digest(tmp_path,[b,a]); assert x==y
    b.write_text('changed'); assert agg.combined_digest(tmp_path,[a,b])!=x
