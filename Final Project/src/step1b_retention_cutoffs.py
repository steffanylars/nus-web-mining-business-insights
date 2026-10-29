"""Describe daily action-age distributions; do not infer an exact platform policy."""
import csv
import numpy as np
import pandas as pd
from config import SRC_USER, RES, FIG, prepare
from data_utils import elapsed_days, split_list

def main():
    prepare()
    types=['CLICK','CART','ORD','FLW']; counts=np.zeros((367,4),dtype=np.int64)
    for_row=0
    with SRC_USER.open(encoding='utf-8',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            acts=np.array(split_list(row['history_type_list']))
            days=elapsed_days([float(x) for x in split_list(row['history_time_list'])],len(acts))
            bins=np.minimum(np.floor(days).astype(int),366)
            for i,t in enumerate(types):
                counts[:,i]+=np.bincount(bins[acts==t],minlength=367)
            for_row+=1
    table=pd.DataFrame(counts,columns=types)
    table.index.name='elapsed_day_floor_366_is_overflow'
    table.to_csv(RES/'retention_daily.csv')
    lines=['Observed daily action-age counts, not verified JD retention policy.',
           f'Users: {for_row:,}. Bins [d,d+1); 366 is >=366 days.',
           'Sharp drops around 7/90 days are consistent with different logging windows.',
           'Rare events remain outside those approximate windows.']
    for i,t in enumerate(types):
        nonzero=np.flatnonzero(counts[:,i])
        lines.append(f'{t}: total={counts[:,i].sum():,}, last occupied day bin={nonzero.max() if len(nonzero) else None}')
    lines.append(f'Orders aged >=365 days: {int(counts[365:,2].sum())}')
    (RES/'retention_cutoffs.txt').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for t in types:
        ax.plot(table.index,table[t],label=t)
    ax.set(yscale='symlog',xlabel='Elapsed days before query (daily bins)',
           ylabel='Observed events',title='Observed logging horizons by action type')
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG/'fig_retention_cutoffs.png',dpi=150); plt.close(fig)

if __name__=='__main__':
    main()
