#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量验证 ETTm1 / ETTm2 的 Context -> Transition 假设。

前提：
    scripts/validate_transition_similarity.py
已经是你当前用于 ETTh1 / ETTh2 的最新版验证脚本，
支持以下参数：
    --csv
    --target
    --history
    --horizon
    --stride
    --topk
    --pairs_per_query
    --output_dir

默认实验设置与 ETTh1 / ETTh2 保持一致：
    target = OT
    history = 96
    horizon = 96
    stride = 12
    topk = 5
    pairs_per_query = 50

运行：
    python scripts/run_validate_ettm.py
"""

import subprocess
import sys
from pathlib import Path


def run_one(dataset_name, output_dir):
    project_root = Path(__file__).resolve().parents[1]
    validator = project_root / "scripts" / "validate_transition_similarity.py"
    csv_path = project_root / "dataset" / "forecasting" / dataset_name

    if not validator.exists():
        raise FileNotFoundError(
            f"找不到验证脚本：{validator}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到数据集：{csv_path}"
        )

    cmd = [
        sys.executable,
        str(validator),
        "--csv", str(csv_path),
        "--target", "OT",
        "--history", "96",
        "--horizon", "96",
        "--stride", "12",
        "--topk", "5",
        "--pairs_per_query", "50",
        "--output_dir", str(project_root / output_dir),
    ]

    print("\n" + "=" * 80)
    print(f"开始验证：{dataset_name}")
    print("=" * 80)
    print("命令：")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"{dataset_name} 验证失败，退出码：{result.returncode}"
        )

    print(f"\n{dataset_name} 验证完成。")
    print(f"结果目录：{project_root / output_dir}")


def main():
    experiments = [
        ("ETTm1.csv", "results_context_transition_ettm1"),
        ("ETTm2.csv", "results_context_transition_ettm2"),
    ]

    for dataset_name, output_dir in experiments:
        run_one(dataset_name, output_dir)

    print("\n" + "=" * 80)
    print("ETTm1 / ETTm2 全部验证完成")
    print("=" * 80)
    print("请重点比较以下指标：")
    print("1. M0_spearman_context_vs_transition")
    print("2. M1_spearman_context_vs_transition")
    print("3. M2_spearman_context_vs_transition")
    print("4. M2_minus_M0_spearman_gain")
    print("5. M0/M1/M2_mean_true_transition_similarity")
    print("6. M0/M1/M2_mean_mse")
    print("7. M0/M1/M2_beats_persistence_ratio")
    print("8. M0/M1/M2_median_skill_vs_persistence")


if __name__ == "__main__":
    main()
