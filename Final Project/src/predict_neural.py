"""Score prepared candidate features and preceding histories without outcome labels."""
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from step5_neural import DHAR,HistoryCF,encode_frames,pack,predict_pack,device
from config import THREADS

def predict(checkpoint,preprocessing,features,history,metadata):
    torch.set_num_threads(THREADS)
    prep=json.loads(Path(preprocessing).read_text(encoding='utf-8'))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    spec=saved['spec']; dev=device()
    model=(HistoryCF(**spec) if spec['variant']=='history_cf' else DHAR(**spec)).to(dev)
    model.load_state_dict(saved['state_dict'])
    frame=features.reset_index(drop=True).copy()
    # Labels never affect inference, even if the caller supplies them.
    frame['label']=0; frame['split']='test'
    encoded,sequences,_=encode_frames(frame,history,metadata,prep)
    packed=pack(encoded,sequences,prep,'test')
    result=frame[['session_id','cand_wid']].copy()
    result['score']=predict_pack(model,packed,len(frame),dev)
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','preprocessing','features','history','metadata','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    output=predict(a.checkpoint,a.preprocessing,pd.read_parquet(a.features),
                   pd.read_parquet(a.history),pd.read_parquet(a.metadata))
    output.to_parquet(a.output,index=False)
    print(f'Scored {len(output):,} candidates',flush=True)

if __name__=='__main__':
    main()
