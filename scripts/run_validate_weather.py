#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证 Weather.csv 的 Context -> Transition 假设。

适配你当前 DAG 项目里的 Weather.csv：
    date,data,cols

当前文件中的通道包括：
    p (mbar), T (degC), ..., Tlog (degC), OT

因此默认：
    endogenous target = OT
    exogenous variables = 除 OT 之外的全部通道

运行：
    python scripts/run_validate_weather.py

如需手动指定目标：
    python scripts/run_validate_weather.py --target OT
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def inspect_channels(csv_path: Path):
    raw = pd.read_csv(csv_path)

    if {"data", "cols"}.issubset(raw.columns):
        channels = [str(x) for x in pd.unique(raw["cols"].dropna())]
        return channels

    return [str(c) for c in raw.columns if c != "date"]


def choose_target(csv_path: Path, manual_target=None):
    channels = inspect_channels(csv_path)

    if manual_target is not None:
        if manual_target not in channels:
            raise ValueError(
                f"指定的 target={manual_target!r} 不存在。\n"
                f"当前可用通道：{channels}"
            )
        return manual_target, channels

    # DAG 项目中的 Weather.csv 实际包含 OT，优先使用 OT
    if "OT" in channels:
        return "OT", channels

    # 兼容其他版本 Weather 数据集：若没有 OT，再尝试 CO2
    co2_candidates = [
        c for c in channels
        if "co2" in c.lower().replace("₂", "2")
    ]
    if co2_candidates:
        return co2_candidates[0], channels

    raise ValueError(
        "无法自动确定 endogenous target。\n"
        f"当前可用通道：{channels}\n"
        "请使用 --target 手动指定。"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        default=None,
        help="手动指定 endogenous target；默认优先使用 OT。"
    )
    parser.add_argument("--history", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pairs_per_query", type=int, default=50)
    parser.add_argument(
        "--output_dir",
        default="results_context_transition_weather"
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    validator = project_root / "scripts" / "validate_transition_similarity.py"
    csv_path = project_root / "dataset" / "forecasting" / "Weather.csv"

    if not validator.exists():
        raise FileNotFoundError(f"找不到验证脚本：{validator}")

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 Weather.csv：{csv_path}")

    target, channels = choose_target(csv_path, args.target)

    exogenous = [c for c in channels if c != target]

    print("\n" + "=" * 80)
    print("Weather Context -> Transition 验证")
    print("=" * 80)
    print(f"数据集：{csv_path}")
    print(f"Endogenous target：{target}")
    print(f"Exogenous variables ({len(exogenous)}):")
    print(exogenous)
    print()

    cmd = [
        sys.executable,
        str(validator),
        "--csv", str(csv_path),
        "--target", target,
        "--history", str(args.history),
        "--horizon", str(args.horizon),
        "--stride", str(args.stride),
        "--topk", str(args.topk),
        "--pairs_per_query", str(args.pairs_per_query),
        "--output_dir", str(project_root / args.output_dir),
    ]

    print("执行命令：")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"Weather 验证失败，退出码：{result.returncode}"
        )

    print("\n" + "=" * 80)
    print("Weather 验证完成")
    print("=" * 80)
    print(f"结果目录：{project_root / args.output_dir}")

    print("\n重点比较：")
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
