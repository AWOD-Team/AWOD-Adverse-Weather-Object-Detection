#!/usr/bin/env python3
"""合成雾生成（只用于训练侧，标定到真实雾的浓度域）。

模型: I = J * t + A * (1 - t),  t = exp(-beta * d)

用法:
  python synthesize_fog.py --root DS --out DS --per-image 3 \
      --beta 1.0 3.5 --A 0.75 1.0 --depth vertical \
      --audit audit.csv [--train-list folds/fold1_train.txt] [--seed 0]

输出:
  DS/images_fog/<stem>_fog<i>.jpg
  DS/labels_fog/<stem>_fog<i>.txt   (直接沿用原标签，几何不变)
  fog_manifest.csv: fog_image,source_image,source_group,beta,A
"""
import argparse, csv, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root):
    root = Path(root)
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMG_EXT and p.is_file()]


def label_for(img, root):
    root, rel = Path(root), img.relative_to(root)
    parts = list(rel.parts)
    for c in ([root / "labels" / Path(*parts[1:]).with_suffix(".txt")] if parts and parts[0] == "images" else []) + \
             [img.with_suffix(".txt")]:
        if c.is_file():
            return c
    return None


def load_depth(img, mode, size=96):
    g = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    h, w = g.shape
    if mode == "flat":
        d = np.full((h, w), 0.5, np.float32)
    elif mode == "vertical":
        grad = np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None].repeat(w, 1)
        d = 0.65 * grad + 0.35 * (1.0 - g)
    else:  # auto: 亮度+模糊的粗糙深度代理
        small = img.convert("L").resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))
        b = np.asarray(small, dtype=np.float32) / 255.0
        d = 0.5 + 0.5 * (1.0 - b)
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return d


def add_fog(img, beta, A, depth, noise, rng, tint=1.0):
    W, H = img.size
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    d = np.asarray(Image.fromarray((depth * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
    t = np.exp(-beta * d)[..., None]
    atm = np.array([A, A, min(1.0, A * tint)], dtype=np.float32)
    out = arr * t + atm * (1.0 - t)
    if noise > 0:
        out = out + np.random.default_rng(rng.randrange(1 << 30)).normal(0, noise, out.shape).astype(np.float32)
    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="输出数据集根目录，默认与 --root 相同")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--beta", nargs=2, type=float, default=[1.0, 3.5])
    ap.add_argument("--A", nargs=2, type=float, default=[0.75, 1.0])
    ap.add_argument("--depth", choices=["vertical", "flat", "auto"], default="vertical")
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", help="audit.csv，用于取 group_id")
    ap.add_argument("--train-list", help="只对这些图片生成（保证雾图只进训练集）")
    ap.add_argument("--suffix", default="_fog")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out) if a.out else root
    rng = random.Random(a.seed)
    (out / "images_fog").mkdir(parents=True, exist_ok=True)
    (out / "labels_fog").mkdir(parents=True, exist_ok=True)

    group_of = {}
    if a.audit and Path(a.audit).is_file():
        with open(a.audit, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                group_of[row["image"]] = row["group_id"]

    images = list_images(root)
    if a.train_list:
        keep = {str(Path(l.strip()).resolve()) for l in Path(a.train_list).read_text().splitlines() if l.strip()}
        images = [p for p in images if str(p.resolve()) in keep]
        print(f"仅对训练清单中的 {len(images)} 张图生成雾图")

    manifest, n_fog = [], 0
    for p in images:
        rel = str(p.relative_to(root))
        gid = group_of.get(rel, p.stem)
        lp = label_for(p, root)
        if not lp:
            print(f"[!] 无标签，跳过 {rel}")
            continue
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[!] 打不开 {rel}: {e}")
            continue
        depth = load_depth(img, a.depth)
        for i in range(a.per_image):
            beta = rng.uniform(*a.beta)
            A = rng.uniform(*a.A)
            tint = rng.uniform(0.97, 1.02)
            fog = add_fog(img, beta, A, depth, a.noise, rng, tint)
            name = f"{p.stem}{a.suffix}{i+1}"
            fog_rel = f"images_fog/{name}.jpg"
            fog.save(str(out / "images_fog" / f"{name}.jpg"), quality=95)
            (out / "labels_fog" / f"{name}.txt").write_text(lp.read_text())
            manifest.append({"fog_image": fog_rel, "source_image": rel,
                             "source_group": gid, "beta": round(beta, 3), "A": round(A, 3)})
            n_fog += 1
    with open(out / "fog_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["fog_image", "source_image", "source_group", "beta", "A"])
        w.writeheader()
        w.writerows(manifest)
    if n_fog:
        bs = [m["beta"] for m in manifest]
        print(f"生成 {n_fog} 张合成雾图（beta {min(bs):.2f}-{max(bs):.2f}），清单: {out/'fog_manifest.csv'}")
    else:
        print("[!] 没有生成任何图")
    return 0


if __name__ == "__main__":
    sys.exit(main())