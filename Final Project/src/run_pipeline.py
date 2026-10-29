"""Run selected corrected stages with a portable Python environment."""
import argparse
import hashlib
import importlib.metadata
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from config import RES,SEED,EPOCHS,HISTORY_LENGTH,prepare,save_json

STAGES={'1':'step1_preprocess_temporal.py','1b':'step1b_retention_cutoffs.py',
    '2':'step2_build_dataset.py','3':'step3_features.py','4':'step4_baselines_lgbm.py',
    '5':'step5_neural.py','6':'step6_diagnostics.py','report':'make_results_summary.py'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--users',type=int,default=20000)
    parser.add_argument('--stages',nargs='+',choices=list(STAGES),default=list(STAGES))
    args=parser.parse_args(); prepare(); src=Path(__file__).parent
    versions={}
    for package in ['numpy','pandas','pyarrow','scipy','scikit-learn','lightgbm','torch']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]='missing'
    status=dict(started_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,
        users=args.users,max_neural_epochs=EPOCHS,history_length=HISTORY_LENGTH,
        dependencies=versions,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(src.glob('*.py'))},stages=[])
    for stage in args.stages:
        cmd=[sys.executable,'-u',str(src/STAGES[stage])]
        if stage=='2': cmd.append(str(args.users))
        status['stages'].append({'stage':stage,'status':'running'})
        save_json(RES/'run_status.json',status)
        result=subprocess.run(cmd,check=False)
        status['stages'][-1]['status']='passed' if result.returncode==0 else 'failed'
        save_json(RES/'run_status.json',status)
        if result.returncode:
            raise SystemExit(result.returncode)
    status['finished_utc']=datetime.now(timezone.utc).isoformat()
    save_json(RES/'run_status.json',status)

if __name__=='__main__':
    main()
