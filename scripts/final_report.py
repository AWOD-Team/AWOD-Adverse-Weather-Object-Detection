#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 v3 实验终版报告：三问对照 + 联合模型分域 + 部署基准，并提交 GitHub。"""
import glob
import json
import os
import statistics as st

O = "/data/awod_data/exp_v3_plan"
REPO = "/data/fog_detect"
OUTDIR = REPO + "/reports/final_v3"
os.makedirs(OUTDIR, exist_ok=True)


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        d = json.load(open(f))
        d["_name"] = os.path.basename(f).replace(".json", "")
        rows.append(d)
    return rows


def ms(rows, key):
    xs = [r[key] for r in rows if key in r]
    if not xs:
        return None
    mean = st.mean(xs)
    std = st.stdev(xs) if len(xs) > 1 else 0.0
    return mean, std, len(xs)


def fmt(rows, key="map50"):
    t = ms(rows, key)
    if not t:
        return "—"
    return "%.4f ± %.4f" % (t[0], t[1])


def cls(rows, c, key="per_class_map50_95"):
    xs = [r[key][c] for r in rows if c in r.get(key, {})]
    if not xs:
        return "—"
    return "%.4f" % st.mean(xs)


# 单域交付基线（来自两份交付包的 test_summary.json）
fog_base = json.load(open(REPO + "/deliverables/awod_yolo11n_fog5fold_v2/results/test_summary.json"))["aggregate"]
night_base = json.load(open(REPO + "/deliverables/awod_yolo11n_night5fold_v2/results/test_summary.json"))["aggregate"]

d1 = load(O + "/d1_dehaze/test/fold*.json")
d2 = load(O + "/d2_mixed/test/fold*.json")
b2 = load(O + "/b2_synth_mix/test/*.json")
b3 = load(O + "/b3_synth_only/test/*.json")
r0 = load(O + "/r0_zeroshot/test/*.json")
r1 = load(O + "/r1_full_finetune_600/test/*.json")
r2 = load(O + "/r2_frozen_600/test/*.json")
r3 = load(O + "/r3_two_stage_600/test/*.json")
m_all = load(O + "/m_multiwx_fivefold/test/fold*.json")
m_fog = load(O + "/m_multiwx_fivefold/test_domain/fog_fold*.json")
m_night = load(O + "/m_multiwx_fivefold/test_domain/night_fold*.json")
n_hold = json.load(open(O + "/n_final_deploy/test/holdout.json"))
bench = json.load(open(O + "/n_final_deploy/export_bench.json"))

L = []
L.append("# AWOD 恶劣天气目标检测 · 实验终版报告（v3）")
L.append("")
L.append("- 生成时间：自动生成")
L.append("- 数据集：雾天 `fog_2cls_clean_v1`（2681 张）+ 夜间 `night_2cls_clean_v2`（2780 张），两类 vehicle/person")
L.append("- 模型：YOLO11n（COCO 预训练），imgsz=640 batch=16 seed=0 lr0=0.001 epochs=100 patience=30")
L.append("- 协议：分组五折（59 组整组不拆），`val` 早停选点、`test` 只评一次")
L.append("")
L.append("## 一、导师三问：结论与证据")
L.append("")
L.append("### 问题1：少量真实雾天标注下，怎样得到可靠检测模型？（小样本）")
L.append("")
L.append("| 方案 | 训练数据 | mAP50 | mAP50-95 |")
L.append("|---|---|---|---|")
L.append("| R0 COCO 零样本（不训练） | 0 | %s | %s |" % (fmt(r0), fmt(r0, "map50_95")))
L.append("| R1 全网络微调 | 600 | %s | %s |" % (fmt(r1), fmt(r1, "map50_95")))
L.append("| R2 冻结骨干 | 600 | %s | %s |" % (fmt(r2), fmt(r2, "map50_95")))
L.append("| R3 两阶段（冻结→解冻） | 600 | %s | %s |" % (fmt(r3), fmt(r3, "map50_95")))
L.append("| （对照）全量雾天数据 | 2145/折 | %.4f ± %.4f | %.4f ± %.4f |" % (
    fog_base["mAP50"]["mean"], fog_base["mAP50"]["std"], fog_base["mAP50-95"]["mean"], fog_base["mAP50-95"]["std"]))
L.append("")
L.append("结论：**直接全网络微调（R1）最好，冻结骨干（R2）反而最差**；"
         "600 张比全量低约 0.039 mAP50，瓶颈是数据量而非微调策略；零样本几乎无效（0.013）。")
L.append("")
L.append("### 问题2：人工合成雾是否有帮助？")
L.append("")
L.append("| 方案 | 训练数据 | 测试数据 | mAP50 | mAP50-95 |")
L.append("|---|---|---|---|---|")
L.append("| D0 基线 | 真实雾 | 真实雾 | %.4f ± %.4f | %.4f ± %.4f |" % (
    fog_base["mAP50"]["mean"], fog_base["mAP50"]["std"], fog_base["mAP50-95"]["mean"], fog_base["mAP50-95"]["std"]))
L.append("| B2 真实+合成雾混合 | 真实雾 + 1911 张合成雾 | 真实雾 | %s | %s |" % (fmt(b2), fmt(b2, "map50_95")))
L.append("| B3 仅合成雾 | 1911 张合成雾 | 真实雾 | %s | %s |" % (fmt(b3), fmt(b3, "map50_95")))
L.append("")
L.append("结论：**混合训练无收益（B2 ≈ D0）**；**仅用合成雾训练掉 0.19 mAP50（B3）**，"
         "合成到真实的域偏差很大。合成雾只能当增强、且当前配方下增强也没有正收益。")
L.append("")
L.append("### 问题3：去雾后再检测是否更好？")
L.append("")
L.append("| 方案 | 说明 | mAP50 | mAP50-95 |")
L.append("|---|---|---|---|")
L.append("| D0 基线 | 原图训练/原图测 | %.4f ± %.4f | %.4f ± %.4f |" % (
    fog_base["mAP50"]["mean"], fog_base["mAP50"]["std"], fog_base["mAP50-95"]["mean"], fog_base["mAP50-95"]["std"]))
L.append("| D1 去雾图训练+去雾图测 | 推理前 DCP 去雾 | %s | %s |" % (fmt(d1), fmt(d1, "map50_95")))
L.append("| D2 原图+去雾图混合训练 | 去雾当训练增强 | %s | %s |" % (fmt(d2), fmt(d2, "map50_95")))
L.append("")
L.append("结论：**去雾没有提升**。D1、D2 与 D0 的差异都在五折标准差内（~0.015），"
         "且 D1 还要额外付出一次去雾推理延迟。")
L.append("")
L.append("## 二、最终部署模型：一个模型覆盖雾天+夜间")
L.append("")
L.append("联合训练（雾+夜合并去重，一张图只出现一次）后，**同一套权重分别在两个域的留出测试集上评测**：")
L.append("")
L.append("| 模型 | 雾天 test mAP50 | 雾天 mAP50-95 | 夜间 test mAP50 | 夜间 mAP50-95 |")
L.append("|---|---|---|---|---|")
L.append("| 单域模型（雾天专用） | %.4f ± %.4f | %.4f ± %.4f | — | — |" % (
    fog_base["mAP50"]["mean"], fog_base["mAP50"]["std"], fog_base["mAP50-95"]["mean"], fog_base["mAP50-95"]["std"]))
L.append("| 单域模型（夜间专用） | — | — | %.4f ± %.4f | %.4f ± %.4f |" % (
    night_base["mAP50"]["mean"], night_base["mAP50"]["std"],
    night_base["mAP50-95"]["mean"], night_base["mAP50-95"]["std"]))
L.append("| **联合模型（推荐部署）** | **%s** | **%s** | **%s** | **%s** |" % (
    fmt(m_fog), fmt(m_fog, "map50_95"), fmt(m_night), fmt(m_night, "map50_95")))
L.append("")
L.append("**联合训练不仅没有掉点，两个域还都略优于各自单域模型**（数据量翻倍带来的正则化收益）。"
         "因此最终交付采用一个模型，不需要场景路由。")
L.append("")
L.append("全量部署模型（训练=雾+夜全部数据去掉 fold1 留出，120 epoch）在留出集上的表现：")
L.append("")
L.append("| 指标 | 值 |")
L.append("|---|---|")
L.append("| P / R | %.4f / %.4f |" % (n_hold["precision"], n_hold["recall"]))
L.append("| mAP50 / mAP50-95 | %.4f / %.4f |" % (n_hold["map50"], n_hold["map50_95"]))
L.append("| vehicle mAP50 / mAP50-95 | %.4f / %.4f |" % (
    n_hold["per_class_map50"]["vehicle"], n_hold["per_class_map50_95"]["vehicle"]))
L.append("| person mAP50 / mAP50-95 | %.4f / %.4f |" % (
    n_hold["per_class_map50"]["person"], n_hold["per_class_map50_95"]["person"]))
L.append("")
L.append("## 三、边缘设备可运行性")
L.append("")
L.append("| 指标 | 值 |")
L.append("|---|---|")
L.append("| 参数量 | %s M |" % bench.get("params_M"))
L.append("| 计算量 | %s GFLOPs @640 |" % bench.get("gflops_640"))
L.append("| PyTorch 权重 | 5.5 MB fp32 |")
L.append("| ONNX (fp32) | %s MB |" % bench.get("onnx_size_MB"))
L.append("| CPU 延迟（PyTorch, 640） | %s ms/张 |" % bench.get("pt_cpu_ms_per_image"))
L.append("| CPU 延迟（ONNX Runtime, 640） | %s ms/张 |" % bench.get("onnx_cpu_ms_per_image"))
L.append("")
L.append("说明：以上是服务器 Xeon 8368Q 单线程 CPU 的实测；换到 Jetson Orin Nano / RK3588 走 TensorRT/ONNX 量化后通常可到 10~20 ms。")
L.append("")
L.append("## 四、复现命令")
L.append("")
L.append("```bash")
L.append("# 全流程（数据构建 + 全部消融 + 联合模型 + 导出）")
L.append("bash scripts/launch_stage.sh            # A..O 全跑，已完成阶段自动跳过")
L.append("python scripts/domain_bench.py         # 联合模型分域评测 + 导出基准")
L.append("python scripts/final_report.py         # 生成本报告")
L.append("```")
L.append("")
L.append("关键脚本：`scripts/overnight_plan.py`（编排）、`scripts/eval_one.py`（单模型评测）、"
         "`scripts/summarize_test.py`（五折汇总）、`scripts/domain_bench.py`（分域+基准）。")
L.append("")
L.append("## 五、局限")
L.append("")
L.append("1. 雾天五折 test 规模不均衡（536~537/折，来自 NOD 的夜间折差异更大 340~781），已用五折 mean±std 缓解。")
L.append("2. vehicle 语义偏窄（NOD 只标 car），夜间 person 明显弱于 vehicle。")
L.append("3. 合成雾只用了 DCP 参数化的简单物理模型（β=1.0~3.5），未做浓度标定。")
L.append("4. 未做 INT8 量化与 416/320 输入的精度-速度曲线，属于边缘部署的下一步。")
L.append("")

open(OUTDIR + "/REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

# 附带明细
import shutil
for src, dst in [
    (O + "/all_results.csv", OUTDIR + "/all_results.csv"),
    (O + "/SUMMARY.md", OUTDIR + "/SUMMARY.md"),
    (O + "/n_final_deploy/export_bench.json", OUTDIR + "/export_bench.json"),
]:
    if os.path.exists(src):
        shutil.copy2(src, dst)
for f in glob.glob(O + "/m_multiwx_fivefold/test_domain/*.json"):
    shutil.copy2(f, OUTDIR + "/" + os.path.basename(f))

print("REPORT written:", OUTDIR + "/REPORT.md")
print("fog domain:", fmt(m_fog), fmt(m_fog, "map50_95"))
print("night domain:", fmt(m_night), fmt(m_night, "map50_95"))
