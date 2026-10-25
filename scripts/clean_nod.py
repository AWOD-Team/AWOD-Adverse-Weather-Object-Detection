#!/usr/bin/env python3
"""Clean the NOD night dataset into a YOLO 2-class dataset.

Rules (all configurable):
  1. 剔除平均亮度过低（肉眼难分辨）的图片：mean_brightness < --min-brightness
  2. 剔除目标数量太少的图片： car+person 框数 < --min-boxes
  3. 剔除疑似自拍/人像摆拍：person 最大框面积占比 >= --portrait-area 且 框数 <= --portrait-max-boxes
  4. bicycle 类整体丢弃（与 RTTS/DAWN 两类口径一致：0=vehicle(car), 1=person）

源数据只读；图片以硬链接方式引用（--img-root 下的 1280 副本）。
同时生成 contact sheet 供人工复核，以及完整剔除台账。
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil
from collections import Counter
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ann-dir', type=Path, default=Path('/data/awod_data/nod_annotations'))
    p.add_argument('--img-root', type=Path, default=Path('/data/awod_data/Nikon750_1280'))
    p.add_argument('--output', type=Path, default=Path('/data/awod_data/processed/nod_2cls_clean_v1'))
    p.add_argument('--min-brightness', type=float, default=10.0)
    p.add_argument('--min-boxes', type=int, default=2)
    p.add_argument('--portrait-area', type=float, default=0.20)
    p.add_argument('--portrait-max-boxes', type=int, default=3)
    p.add_argument('--contact-size', type=int, default=60)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def mean_luma(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return float(im.mean()) if im is not None else -1.0

def main():
    a = parse_args()
    out = a.output
    if out.exists():
        if not a.overwrite:
            raise SystemExit(f'ABORT: {out} 已存在（加 --overwrite 覆盖）')
        shutil.rmtree(out)
    (out/'images').mkdir(parents=True)
    (out/'labels').mkdir(parents=True)
    (out/'review').mkdir()

    from concurrent.futures import ThreadPoolExecutor
    import cv2

    # 1) 读三个 COCO json，收集每图标注
    items = []
    for split in ('train', 'val', 'test'):
        d = json.load(open(a.ann_dir/f'NOD_Nikon750_{split}.json'))
        cats = {c['id']: c['name'] for c in d['categories']}
        by = {}
        for ann in d['annotations']:
            by.setdefault(ann['image_id'], []).append(ann)
        for im in d['images']:
            items.append(dict(split=split, name=im['file_name'], W=im['width'], H=im['height'],
                              anns=by.get(im['id'], []), cats=cats))
    print(f'标注图片总数: {len(items)}')

    # 2) 亮度
    def _l(it):
        return mean_luma(a.img_root/it['name'])
    with ThreadPoolExecutor(max_workers=8) as ex:
        lum = list(ex.map(_l, items, chunksize=32))
    for it, l in zip(items, lum):
        it['brightness'] = l

    # 3) 逐图判定
    kept, rejected = [], []
    for it in items:
        W, H = it['W'], it['H']
        car = person = 0
        max_p = 0.0
        lines = []
        for ann in it['anns']:
            nm = it['cats'][ann['category_id']]
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            x1, y1, x2, y2 = max(0.0, x), max(0.0, y), min(W, x+w), min(H, y+h)
            bw, bh = x2-x1, y2-y1
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (x1+bw/2)/W, (y1+bh/2)/H
            nw, nh = bw/W, bh/H
            if nm == 'car':
                car += 1; lines.append(f'0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
            elif nm == 'person':
                person += 1; lines.append(f'1 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}')
                max_p = max(max_p, (bw*bh)/(W*H))
            # bicycle 丢弃
        nbox = car + person
        it.update(car=car, person=person, n_box=nbox, max_person_area=round(max_p, 6), yolo=lines)
        reason = ''
        if it['brightness'] < a.min_brightness:
            reason = 'too_dark'
        elif nbox < a.min_boxes:
            reason = 'too_few_objects'
        elif max_p >= a.portrait_area and nbox <= a.portrait_max_boxes:
            reason = 'portrait_or_posed'
        if reason:
            it['reason'] = reason; rejected.append(it)
        else:
            kept.append(it)

    # 4) 落盘：硬链接图片 + 写标签
    for it in kept:
        src = a.img_root/it['name']
        dst = out/'images'/it['name']
        if not dst.exists():
            try: os.link(src, dst)
            except OSError: shutil.copy2(src, dst)
        (out/'labels'/(Path(it['name']).stem + '.txt')).write_text(
            '\n'.join(it['yolo']) + ('\n' if it['yolo'] else ''), encoding='utf-8')

    # 5) 台账
    def write_csv(path, rows, cols):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({c: r.get(c, '') for c in cols})
    base = ['name','split','brightness','car','person','n_box','max_person_area']
    write_csv(out/'selection.csv', kept, base)
    write_csv(out/'exclusion_ledger.csv', rejected, base+['reason'])
    (out/'classes.yaml').write_text('names:\n  0: vehicle\n  1: person\nnc: 2\n', encoding='utf-8')
    (out/'all.txt').write_text(''.join(f'{out}/images/{it["name"]}\n' for it in kept), encoding='utf-8')

    # 6) contact sheets（剔除样本 + 保留抽样）
    ORIG = Path('/data/awod_data/Nikon750_1280')
    def sheet(rows, tag, limit=60, cols=10, size=a.contact_size):
        rows = rows[:limit]
        if not rows: return
        rows_n = int(np.ceil(len(rows)/cols))
        canvas = np.full((rows_n*size, cols*size, 3), 40, np.uint8)
        for i, it in enumerate(rows):
            im = cv2.imread(str(ORIG/it['name']))
            if im is None: continue
            im = cv2.resize(im, (size, size))
            r, c = divmod(i, cols)
            canvas[r*size:(r+1)*size, c*size:(c+1)*size] = im
            cv2.putText(canvas, f"{it['n_box']}b {it['brightness']:.0f}L", (c*size+2, r*size+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
        cv2.imwrite(str(out/'review'/f'{tag}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for reason in ('too_dark','too_few_objects','portrait_or_posed'):
        sheet([r for r in rejected if r['reason']==reason], f'rejected_{reason}')
    sheet(sorted(kept, key=lambda r: r['brightness'])[:30], 'kept_darkest_30')
    sheet(kept[:30], 'kept_sample_30')

    # 7) SHA256SUMS
    lines = []
    for f in ('classes.yaml','all.txt','selection.csv','exclusion_ledger.csv'):
        lines.append(f'{hashlib.sha256((out/f).read_bytes()).hexdigest()}  {f}')
    (out/'SHA256SUMS').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    # 8) 统计
    print()
    print('=== 清洗结果 ===')
    print(f'  原始标注图片 : {len(items)}')
    print(f'  保留         : {len(kept)}  ({100*len(kept)/len(items):.1f}%)')
    print(f'  剔除         : {len(rejected)}')
    for r, n in Counter(x["reason"] for x in rejected).most_common():
        print(f'      {r:20s}: {n}')
    print(f'  保留车框     : {sum(x["car"] for x in kept)}  人框: {sum(x["person"] for x in kept)}  (丢弃 bicycle)')
    kb = np.array([x['brightness'] for x in kept])
    print(f'  保留图亮度   : min={kb.min():.1f} p5={np.percentile(kb,5):.1f} 中位={np.median(kb):.1f}')
    print(f'  保留框数分布 : 2框={sum(1 for x in kept if x["n_box"]==2)} 3框={sum(1 for x in kept if x["n_box"]==3)} >=4框={sum(1 for x in kept if x["n_box"]>=4)}')

if __name__ == '__main__':
    main()
