"""Write a current source/data inventory without machine-specific paths."""
from config import BASE,RES,prepare

def main():
    prepare()
    files=sorted((BASE/'src').glob('*.py'))
    lines=['# Dataset and source code inventory','',
        'Raw data: JDSearch user_behavior_data.txt and product_meta_data.txt.',
        'Obtain from https://github.com/rucliujn/JDsearch. Raw files are not redistributed.',
        'Set JD_USER_TXT and JD_PRODUCT_TXT; raw metadata is streamed directly.',
        'Corrected outputs: artifacts/corrected. See README.md for reproduction.','',
        '| Source | Bytes |','|---|---:|']
    lines.extend(f'| src/{p.name} | {p.stat().st_size:,} |' for p in files)
    (RES/'SOURCE_LISTING.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    main()
