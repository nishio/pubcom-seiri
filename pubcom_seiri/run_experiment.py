#!/usr/bin/env python3
"""
run_experiment.py: 複数フェーズを連携して実行するスクリプト

機能:
- フェーズ1: CSVデータの読み込みと重複除去、文字ベースの類似度によるグループ化
- フェーズ2: 埋め込みベースの処理とクラスタリング
- フェーズ3: クラスタリング結果の詳細分析（tool2）
- 結果を指定されたディレクトリに出力
"""

import argparse
import os
import subprocess
import sys
from typing import List, Dict, Any


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="CSVデータのクラスタリングと詳細分析を実行"
    )
    parser.add_argument("--input", required=True, help="入力CSVファイルのパス")
    parser.add_argument("--output", required=True, help="出力ディレクトリのパス")
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="分析対象とするCSVデータの末尾n件 (default: 10000)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="クラスタリングの距離閾値 (default: 0.4)",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=50.0,
        help="文字ベースの類似度閾値（%） (default: 50.0)",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="使用する埋め込みモデル名",
    )
    return parser.parse_args()


def run_phase1(args: argparse.Namespace) -> str:
    """フェーズ1: CSVデータの読み込みと重複除去、文字ベースの類似度によるグループ化"""
    print("Running Phase 1: CSV reading and duplicate removal...")

    # フェーズ1の出力ディレクトリを作成
    phase1_output_dir = os.path.join(args.output, "phase1_output")
    os.makedirs(phase1_output_dir, exist_ok=True)

    cmd = [
        "python3",
        "pubcom_seiri/phase1.py",
        "--input",
        args.input,
        "--output",
        phase1_output_dir,
        "--limit",
        str(args.limit),
        "--similarity",
        str(args.similarity),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Phase 1 completed successfully. Results saved to {phase1_output_dir}")
        return phase1_output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running Phase 1: {e}", file=sys.stderr)
        sys.exit(1)


def run_phase2(args: argparse.Namespace, phase1_output_dir: str) -> str:
    """フェーズ2: 埋め込みベースの処理とクラスタリング"""
    print("Running Phase 2: Embedding-based processing...")

    # フェーズ2の出力ディレクトリを作成
    phase2_output_dir = os.path.join(args.output, "phase2_output")
    os.makedirs(phase2_output_dir, exist_ok=True)

    cmd = [
        "python3",
        "pubcom_seiri/phase2.py",
        "--input",
        phase1_output_dir,
        "--output",
        phase2_output_dir,
        "--threshold",
        str(args.threshold),
        "--model",
        args.model,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Phase 2 completed successfully. Results saved to {phase2_output_dir}")
        return phase2_output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running Phase 2: {e}", file=sys.stderr)
        sys.exit(1)


def run_phase3(args: argparse.Namespace, phase2_output_dir: str) -> str:
    """フェーズ3: クラスタリング結果の詳細分析（tool2）"""
    print("Running Phase 3: Detailed analysis...")

    # フェーズ3の出力ディレクトリを作成
    phase3_output_dir = os.path.join(args.output, "phase3_output")
    os.makedirs(phase3_output_dir, exist_ok=True)

    cmd = [
        "python3",
        "pubcom_seiri/tool2.py",
        "--input",
        phase2_output_dir,
        "--original",
        args.input,
        "--threshold",
        str(args.threshold),
        "--output",
        phase3_output_dir,
        "--model",
        args.model,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Phase 3 completed successfully. Results saved to {phase3_output_dir}")
        return phase3_output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running Phase 3: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()

    # 出力ディレクトリを作成
    os.makedirs(args.output, exist_ok=True)

    phase1_output_dir = run_phase1(args)

    phase2_output_dir = run_phase2(args, phase1_output_dir)

    phase3_output_dir = run_phase3(args, phase2_output_dir)

    print("Experiment completed successfully!")
    print(f"Phase 1 results: {phase1_output_dir}")
    print(f"Phase 2 results: {phase2_output_dir}")
    print(f"Phase 3 results: {phase3_output_dir}")


if __name__ == "__main__":
    main()
