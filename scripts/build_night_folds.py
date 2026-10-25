#!/usr/bin/env python3
"""为夜间主数据集构建按来源分组的 k 折划分（train/val/test）。

分组规则：
  nod__*         -> 按 EXIF DateTime 的拍摄日（同一天连续拍摄，必须同折）
  rtts__hvNN_*   -> hv 视频源整段绑定
  其它 rtts__*   -> 城市_来源（去尾部编号）
  dawn__*        -> 天气序列名（去尾部编号）
  最后按感知哈希近重复对合并组

分配：组内图片不拆，按框数贪心装箱到 k 折（每折的 test = 该折的组）；
每折再从其余组里按组挑出约 12.5% 作为 val（早停/选模），剩下为 train。
"""
from __future__ import annotations
import argparse, json, os, re, datetime as dt
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ds', type=Path, default=Path('/data/awod_data/processed/night_2cls_clean_v2'))
    p.add_argument('--dup-json', type=Path, default=Path('/data/fog_detect/reports/night_2cls_v2_dup.dup.json'))
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--val-ratio', type=float, default=0.125)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def exif_day(path):
    from PIL import Image
    try:
        ex = Image.open(path).getexif()
        s = ex.get(306) or ex.get(36867)
        return dt.datetime.strptime(s, '%Y:%m:%d %H:%M:%S').date().isoformat()
    except Exception:
        return 'unknown'

def group_of(name, exif_days):
    stem = Path(name).stem
    if stem.startswith('nod__'):
        return 'nod_date_' + exif_days.get(name, 'unknown')
    if stem.startswith('rtts__'):
        s = stem[len('rtts__'):]
        m = re.match(r'^(hv\d+)_\d+$', s)
        if m: return 'rtts_video_' + m.group(1)
        m2 = re.match(r'^(.*?)_\d+$', s)
        return 'rtts_' + (m2.group(1) if m2 else s)
    if stem.startswith('dawn__'):
        s = stem[len('dawn__'):]
        m = re.match(r'^(.*?)-\d+$', s)
        return 'dawn_' + (m.group(1) if m else s)
    return 'other_' + stem

def box_counts(ds, names):
    out = {}
    for n in names:
        p = ds/'labels'/(Path(n).stem + '.txt')
        v = q = 0
        if p.is_file():
            for line in open(p):
                f = line.split()
                if not f: continue
                if f[0] == '0': v += 1
                elif f[0] == '1': q += 1
        out[n] = (v, q)
    return out

def main():
    a = parse_args()
    ds = a.ds
    names = sorted(os.listdir(ds/'images'))
    print(f'图片: {len(names)}')

    # EXIF 拍摄日（仅 nod__）
    exif_days = {}
    nod = [n for n in names if n.startswith('nod__')]
    from PIL import Image
    for n in nod:
        exif_days[n] = exif_day(ds/'images'/n)
    print(f'NOD {len(nod)} 张, 拍摄日 {len(set(exif_days.values()))} 个')

    bc = box_counts(ds, names)

    # 初始分组
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry

    gmap = {n: group_of(n, exif_days) for n in names}
    by_g = defaultdict(list)
    for n, g in gmap.items(): by_g[g].append(n)
    for g, members in by_g.items():
        for m in members[1:]: union(members[0], m)

    # phash 近重复并组
    if a.dup_json.is_file():
        d = json.load(open(a.dup_json))
        pairs = d.get('pairs', [])
        for pr in pairs:
            if len(pr) >= 2 and pr[0] in parent and pr[1] in parent:
                union(pr[0], pr[1])
        print(f'phash 对 {len(pairs)} 个已并入组')

    # 汇总组
    groups = defaultdict(lambda: {'names': [], 'box': 0})
    for n in names:
        r = find(n)
        groups[r]['names'].append(n)
        v, q = bc[n]; groups[r]['box'] += v + q
    glist = sorted(groups.items(), key=lambda kv: -len(kv[1]['names']))
    print(f'最终组数: {len(glist)}, 最大组 {len(glist[0][1]["names"])} 张')
    print('  组规模 top8:', [(len(g["names"])) for _, g in glist[:8]])

    # 贪心装箱到 k 折（按框数均衡）
    k = a.k
    folds = [{'id': i+1, 'groups': [], 'box': 0, 'n': 0} for i in range(k)]
    for gid, g in sorted(glist, key=lambda kv: -kv[1]['box']):
        t = min(folds, key=lambda f: (f['box'], f['n']))
        t['groups'].append(gid); t['box'] += g['box']; t['n'] += len(g['names'])

    # 每折：test = 本折组；val 从其余组里按组挑
    out = ds/'folds5_grouped_v1'
    if out.exists():
        if not a.overwrite: raise SystemExit(f'ABORT: {out} 已存在')
        import shutil; shutil.rmtree(out)
    (out).mkdir(parents=True)

    summary = {'seed': a.seed, 'k': k, 'images': len(names), 'groups': len(glist),
               'val_ratio': a.val_ratio, 'groups_policy': 'EXIF day / hv video / city-source / dawn sequence + phash', 'folds': []}
    gmanifest, smanifest = [], []

    for f in folds:
        test_g = set(f['groups'])
        test_names = [n for gid in f['groups'] for n in groups[gid]['names']]
        rest = [(gid, groups[gid]) for gid, _ in glist if gid not in test_g]

        # val 目标框数
        rest_box = sum(g['box'] for _, g in rest)
        val_target = rest_box * a.val_ratio / (1.0 - a.val_ratio)
        rng = np.random.RandomState(a.seed + f['id'])
        order = [gid for gid, _ in sorted(rest, key=lambda kv: -kv[1]['box'])]
        rng.shuffle(order)  # 同规模组之间随机化，避免总选同样的组
        order.sort(key=lambda gid: -groups[gid]['box'])
        val_g, val_box = set(), 0
        for gid in order:
            if val_box >= val_target: break
            if val_box + groups[gid]['box'] > val_target * 1.4: continue
            val_g.add(gid); val_box += groups[gid]['box']

        val_names = [n for gid in val_g for n in groups[gid]['names']]
        train_names = [n for gid, g in rest if gid not in val_g for n in g['names']]

        fd = out/f'fold{f["id"]}'
        fd.mkdir()
        for tag, lst in (('train', train_names), ('val', val_names), ('test', test_names)):
            (fd/f'{tag}.txt').write_text(''.join(f'{ds}/images/{n}\n' for n in sorted(lst)), encoding='utf-8')
        (fd/'data.yaml').write_text(
            f'path: {ds}\ntrain: {fd}/train.txt\nval: {fd}/val.txt\ntest: {fd}/test.txt\n'
            'names:\n  0: vehicle\n  1: person\n', encoding='utf-8')

        cov = Counter()
        for n in train_names: cov['train'] += 1
        for n in val_names: cov['val'] += 1
        for n in test_names: cov['test'] += 1
        info = {'fold': f['id'], 'train': len(train_names), 'val': len(val_names), 'test': len(test_names),
                'test_groups': len(f['groups']), 'val_groups': len(val_g),
                'test_src': dict(Counter(n.split('__')[0] for n in test_names))}
        summary['folds'].append(info)
        print(f"fold{f['id']}: train={len(train_names):4d} val={len(val_names):3d} test={len(test_names):4d} "
              f"(test 组={len(f['groups'])}, test 来源={info['test_src']})")

        for n in test_names: gmanifest.append({'name': n, 'group_id': find(n), 'test_fold': f['id']})
        for tag, lst in (('train', train_names), ('val', val_names), ('test', test_names)):
            for n in lst: smanifest.append({'name': n, 'fold': f['id'], 'split': tag, 'group_id': find(n)})

    # 泄漏审计
    g2f = defaultdict(set)
    for r in smanifest:
        if r['split'] == 'test': g2f[r['group_id']].add(r['fold'])
    leaks = {g: s for g, s in g2f.items() if len(s) > 1}
    tested = Counter(r['name'] for r in smanifest if r['split'] == 'test')
    dup_test = {n: c for n, c in tested.items() if c > 1}
    summary['audit'] = {'group_leaks': len(leaks), 'images_tested_once': len(tested),
                        'images_tested_more_than_once': len(dup_test), 'total_images': len(names)}
    print()
    print('=== 泄漏审计 ===')
    print('  组跨折数:', len(leaks))
    print('  当 test 的图片数:', len(tested), '/', len(names), '(应为相等)')
    print('  被测试多次:', len(dup_test))

    (out/'fold_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    def wc(path, rows, cols):
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    wc(out/'group_manifest.csv', gmanifest, ['name','group_id','test_fold'])
    wc(out/'split_manifest.csv', smanifest, ['name','fold','split','group_id'])

if __name__ == '__main__':
    main()
