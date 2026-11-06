"""The notebook must still open historical test-only saved evidence."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eval_utils import evaluate


class NotebookEvidence(unittest.TestCase):
    def test_historical_results_do_not_require_unavailable_split_metrics(self):
        notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/Step4_Baselines_LambdaMART.ipynb').read_text(encoding='utf-8'))
        cells=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
        table=next(s for s in cells if s.startswith('res = json.loads'))
        verify=next(s for s in cells if s.startswith('pred = pd.read_parquet'))
        models=['Random','Popularity','Content/Relevance-only','SVD(train-only fold-in)','LambdaMART(features)']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            metrics={m:{'NDCG@10':1.,'MRR@200':1.,'P@1':1.,'HR@10':1.} for m in models}
            (root/'results_baselines_lgb.json').write_text(json.dumps(metrics))
            (root/'baseline_protocol.json').write_text(json.dumps({'best_iteration':37,
                'svd_train_history_item_coverage':.5,'lgb_parameters':{},'evaluation':'test',
                'query_counts':{m:{'queries_with_positive':1} for m in models}}))
            frame=pd.DataFrame({'session_id':[0,0],'cand_wid':['a','b'],'label':[1,0]})
            for col in ['random_score','pop_wid','content_score','svd_score','lgb_score']:
                frame[col]=[2.,1.]
            frame.to_parquet(root/'test_pred_lgb.parquet',index=False)
            scope={'RES':root,'OUT':root,'pd':pd,'np':np,'json':json,'SEED':42,'display':lambda x:None,'evaluate':evaluate}
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(table,'notebook-results','exec'),scope)
                exec(compile(verify,'notebook-verification','exec'),scope)
            self.assertIn('unavailable',output.getvalue().lower())


if __name__=='__main__':
    unittest.main()
