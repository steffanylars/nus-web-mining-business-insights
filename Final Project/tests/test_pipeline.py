"""Regression checks for the audit's concrete failure modes."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from data_utils import tok,elapsed_days,short_mask,select_sequence
from step2_build_dataset import build_records
from eval_utils import evaluate,lightgbm_metric
from step4_baselines_lgbm import fold_in_svd
from step6_diagnostics import positive_rank_records
from step5_neural import DHAR,HistoryCF,listwise,brand_bucket,fit_preprocessing,encode_frames,pack

class PipelineRegression(unittest.TestCase):
    def test_actual_delimiter_and_missing_query(self):
        self.assertEqual(tok('12\x1832\x1856'),{'12','32','56'})
        self.assertEqual(tok('-1'),set())
        self.assertEqual(tok(None),set())
        self.assertEqual(len(tok('12\x1832')&tok('7\x1832')),1)

    def test_timeline_and_full_order_retention(self):
        np.testing.assert_array_equal(elapsed_days([0,12,4,5],3)*86400,[21,9,5])
        with self.assertRaises(ValueError):
            elapsed_days([0,1],2)
        row=dict(query='12\x1832',candidate_wid_list='a_b',candidate_label_list='1_0',
            history_wid_list='_'.join(['old']+['new']*70),
            history_type_list='_'.join(['ORD']+['CLICK']*70),
            history_qry_list='_'.join(['-1']*71),history_time_list='_'.join(['0']+['1']*71))
        cand,hist,sess=build_records([row],[0])
        self.assertEqual(len(hist),71)
        self.assertEqual(hist.iloc[0]['type'],'ORD')
        self.assertEqual(sess.iloc[0].source_row_id,0)

    def test_horizons_and_sequence_endpoints(self):
        hist=pd.DataFrame({'type':['CLICK','CLICK','CART','CART','ORD','FLW'],
                           'elapsed_days':[7,8,90,91,1,1]})
        self.assertEqual(short_mask(hist).tolist(),[True,False,True,False,False,False])
        seq=select_sequence(np.arange(100),20)
        self.assertEqual((seq[0],seq[-1],len(seq)),(0,99,20))

    def test_order_independent_ties_and_all_negative_policy(self):
        df=pd.DataFrame({'session_id':[0,0,0,1,1],
            'cand_wid':['a','b','c','d','e'],'label':[1,0,2,0,0],'score':[0.,0.,0.,0.,0.]})
        expected=evaluate(df,'score')
        self.assertEqual(expected[1],1)
        self.assertEqual(expected,evaluate(df.sample(frac=1,random_state=17),'score'))
        metric=lightgbm_metric(df)(df.score.to_numpy(),None)
        self.assertAlmostEqual(metric[1],expected[0]['NDCG@10'])
        df.loc[0,'score']=np.nan
        with self.assertRaises(ValueError):
            evaluate(df,'score')

    def test_svd_shared_factors_ignore_heldout_events(self):
        hist=pd.DataFrame({'session_id':[0,0,1,1,2,2,3],
                           'wid':['a','b','a','c','b','c','heldout']})
        items,svd,u=fold_in_svd(hist,{0,1,2},4,dimensions=2)
        changed=pd.concat([hist,pd.DataFrame({'session_id':[3],'wid':['a']})],ignore_index=True)
        items2,svd2,u2=fold_in_svd(changed,{0,1,2},4,dimensions=2)
        self.assertNotIn('heldout',items)
        self.assertEqual(items,items2)
        np.testing.assert_allclose(svd.components_,svd2.components_)
        self.assertGreater(np.abs(u2[3]).sum(),0)

    def test_attention_empty_history_and_loss_masking(self):
        model=DHAR(4,20,2).eval()
        num=torch.zeros(2,3,2); c3=torch.ones(2,3,dtype=torch.long)
        zeros=torch.zeros(2,4,dtype=torch.long)
        score=model(num,c3,c3,c3,zeros,zeros,zeros)
        self.assertTrue(torch.isfinite(score).all())
        context=model.attend(model.ec3(c3),zeros,model.ps)
        self.assertTrue(torch.equal(context,torch.zeros_like(context)))
        loss=listwise(score,torch.zeros(2,3,dtype=torch.long),torch.ones(2,3,dtype=torch.bool))
        self.assertEqual(float(loss.detach()),0.)
        loss.backward()

    def test_cold_positive_rank_keeps_original_slate(self):
        pred=pd.DataFrame({'session_id':[0]*5,'cand_wid':['a','b','c','d','cold'],
            'label':[0,0,0,0,1],'score':[5.,4.,3.,2.,1.]})
        ranks=positive_rank_records(pred,'score',{'a','b','c','d'},{'a','b'})
        self.assertEqual(int(ranks.iloc[0]['rank']),5)
        self.assertFalse(bool(ranks.iloc[0].training_history_seen))
        self.assertFalse(bool(ranks.iloc[0].training_candidate_seen))

    def test_history_cf_has_no_user_id_parameters(self):
        model=HistoryCF(8).eval()
        item=torch.tensor([[1,2],[1,2]]); hist=torch.tensor([[3,4],[3,4]])
        out=model(torch.zeros(2,2,1),item,item,item,hist,hist,hist)
        torch.testing.assert_close(out[0],out[1])
        self.assertFalse(any('user' in name for name,_ in model.named_parameters()))

    def test_preprocessing_is_train_only_and_checkpoint_reload(self):
        feat=pd.DataFrame({'session_id':[0,0,1,1,2,2],'cand_wid':['a','b','a','c','a','c'],
            'label':[1,0,1,0,0,1],'split':['train','train','val','val','test','test'],
            'feature':[1.,3.,100.,200.,300.,400.]})
        hist=pd.DataFrame({'session_id':[0,1,2],'pos':[0,0,0],'wid':['a','c','c'],
                           'type':['ORD','CART','CLICK'],'elapsed_days':[100.,10.,1.]})
        meta=pd.DataFrame({'wid':['a','b','c'],'cate_id_3':['x','x','heldout'],'brand_id':['b','b','b']})
        prep=fit_preprocessing(feat,hist,meta,20)
        self.assertEqual(prep['mean'],[2.])
        self.assertNotIn('heldout',prep['category_ids'])
        self.assertNotIn('c',prep['item_ids'])
        enc,seq,cov=encode_frames(feat,hist,meta,prep)
        pk=pack(enc,seq,prep,'test')
        self.assertEqual(pk['mask'].sum().item(),2)
        self.assertEqual(pk['c3'][0,:2].tolist(),[1,0])
        model=DHAR(2,20000,1).eval()
        duplicate=DHAR(2,20000,1).eval(); duplicate.load_state_dict(model.state_dict())
        args=[pk[k] for k in ['num','c3','bi','ii','short','order','cf']]
        torch.testing.assert_close(model(*args),duplicate(*args))
        self.assertEqual(brand_bucket('brand'),brand_bucket('brand'))

if __name__=='__main__':
    unittest.main()
