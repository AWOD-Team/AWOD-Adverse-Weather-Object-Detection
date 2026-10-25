#!/usr/bin/env python3
"""把已清洗的夜间数据集合并成夜间主数据集（YOLO 两类）。

来源：
  night_2cls_clean_v1  (RTTS 夜间 + DAWN 夜间，文件名已带 rtts__ / dawn__ 前缀)
  nod_2cls_clean_v1    (NOD 夜间，文件名无前缀 -> 加 nod__)
图片以硬链接引用，标签复制。生成 manifest / all.txt / classes.yaml / SHA256SUMS。
五折划分由 build_night_folds.py 单独完成。
"""
from __future__ import annotations
import argparse, csv, hashlib, os, shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/awod_data/processed'))
    p.add_argument('--output', type=Path, default=Path('/data/awod_data/processed/night_2cls_clean_v2'))
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def read_csv_map(path, key):
    if not path.is_file(): return {}
    return {r[key]: r for r in csv.DictReader(open(path, encoding='utf-8-sig'))}

def main():
    a = parse_args()
    out = a.output
    if out.exists():
        if not a.overwrite: raise SystemExit(f'ABORT: {out} 已存在')
        shutil.rmtree(out)
    (out/'images').mkdir(parents=True)
    (out/'labels').mkdir(parents=True)

    n_meta = read_csv_map(a.root/'night_2cls_clean_v1'/'manifest.csv', 'merged_name')
    d_meta = read_csv_map(a.root/'nod_2cls_clean_v1'/'selection.csv', 'name')

    rows = []
    # --- 来源 1: night_2cls_clean_v1 (已带前缀) ---
    src = a.root/'night_2cls_clean_v1'
    for name in sorted(os.listdir(src/'images')):
        stem, _ = os.path.splitext(name)
        source = 'rtts' if stem.startswith('rtts__') else ('dawn' if stem.startswith('dawn__') else 'unknown')
        m = n_meta.get(name, {})
        rows.append(dict(merged_name=name, source=source,
                         original_name=m.get('original_name', stem.split('__',1)[-1]),
                         brightness=m.get('mean_brightness',''), vehicle=m.get('vehicle',''),
                         person=m.get('person',''), boxes=m.get('boxes',''),
                         origin_split='', flag='not_in_clean_set' if m.get('in_clean_set')=='False' else '',
                         src_path=str(src/'images'/name), src_label=str(src/'labels'/(stem+'.txt'))))
    # --- 来源 2: nod_2cls_clean_v1 (加 nod__ 前缀) ---
    src = a.root/'nod_2cls_clean_v1'
    for name in sorted(os.listdir(src/'images')):
        stem, ext = os.path.splitext(name)
        new = f'nod__{name}'
        m = d_meta.get(name, {})
        rows.append(dict(merged_name=new, source='nod', original_name=name,
                         brightness=m.get('brightness',''), vehicle=m.get('car',''),
                         person=m.get('person',''), boxes=m.get('n_box',''),
                         origin_split=m.get('split',''), flag='',
                         src_path=str(src/'images'/name), src_label=str(src/'labels'/(stem+'.txt'))))

    # --- 落盘 ---
    missing = []
    for r in rows:
        dst_img = out/'images'/r['merged_name']
        dst_lbl = out/'labels'/(Path(r['merged_name']).stem + '.txt')
        if not os.path.exists(r['src_label']):
            missing.append(r['merged_name']); continue
        try: os.link(r['src_path'], dst_img)
        except OSError: shutil.copy2(r['src_path'], dst_img)
        shutil.copy2(r['src_label'], dst_lbl)

    cols = ['merged_name','source','original_name','brightness','vehicle','person','boxes','origin_split','flag']
    with open(out/'manifest.csv','w',newline='',encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow({c: r.get(c,'') for c in cols})
    (out/'all.txt').write_text(''.join(f"{out}/images/{r['merged_name']}\n" for r in rows), encoding='utf-8')
    (out/'classes.yaml').write_text('names:\n  0: vehicle\n  1: person\nnc: 2\n', encoding='utf-8')
    lines = [f"{hashlib.sha256((out/f).read_bytes()).hexdigest()}  {f}"
             for f in ('classes.yaml','all.txt','manifest.csv')]
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    from collections import Counter
    print('=== 合并完成 ===')
    print('  合计图片 :', len(rows))
    for s, n in Counter(r['source'] for r in rows).most_common():
        print(f'    {s:6s}: {n}')
    print('  vehicle框:', sum(int(r['vehicle']) for r in rows if str(r['vehicle']).isdigit()))
    print('  person 框:', sum(int(r['person'])  for r in rows if str(r['person']).isdigit()))
    print('  标记 not_in_clean_set:', sum(1 for r in rows if r['flag']))
    print('  缺标签:', missing)

if __name__ == '__main__':
    main()
