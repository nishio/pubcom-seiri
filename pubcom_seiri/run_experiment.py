#!/usr/bin/env python3
"""
run_experiment.py: tool1とtool2を連携して実行するスクリプト

機能:
- CSVデータを読み込み、tool1でクラスタリングを実行
- tool1の結果をtool2で詳細分析
- 結果を指定されたディレクトリに出力
"""

import argparse
import os
import subprocess
import sys
from typing import List, Dict, Any

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='CSVデータのクラスタリングと詳細分析を実行')
    parser.add_argument('--input', required=True, help='入力CSVファイルのパス')
    parser.add_argument('--output', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--limit', type=int, default=10000, help='分析対象とするCSVデータの末尾n件 (default: 10000)')
    parser.add_argument('--threshold', type=float, default=0.4, help='クラスタリングの距離閾値 (default: 0.4)')
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='使用する埋め込みモデル名')
    return parser.parse_args()

def run_tool1(args: argparse.Namespace) -> str:
    """tool1を実行する"""
    print("Running tool1...")
    
    # tool1の出力ディレクトリを作成
    tool1_output_dir = os.path.join(args.output, "tool1_output")
    os.makedirs(tool1_output_dir, exist_ok=True)
    
    # tool1コマンドを構築
    cmd = [
        "python3", "pubcom_seiri/tool1.py",
        "--input", args.input,
        "--output", tool1_output_dir,
        "--limit", str(args.limit),
        "--threshold", str(args.threshold),
        "--model", args.model
    ]
    
    # tool1を実行
    try:
        subprocess.run(cmd, check=True)
        print(f"tool1 completed successfully. Results saved to {tool1_output_dir}")
        return tool1_output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running tool1: {e}", file=sys.stderr)
        sys.exit(1)

def run_tool2(args: argparse.Namespace, tool1_output_dir: str) -> str:
    """tool2を実行する"""
    print("Running tool2...")
    
    # tool2の出力ディレクトリを作成
    tool2_output_dir = os.path.join(args.output, "tool2_output")
    os.makedirs(tool2_output_dir, exist_ok=True)
    
    # tool2コマンドを構築
    cmd = [
        "python3", "pubcom_seiri/tool2.py",
        "--input", tool1_output_dir,
        "--original", args.input,
        "--threshold", str(args.threshold),
        "--output", tool2_output_dir,
        "--model", args.model
    ]
    
    # tool2を実行
    try:
        subprocess.run(cmd, check=True)
        print(f"tool2 completed successfully. Results saved to {tool2_output_dir}")
        return tool2_output_dir
    except subprocess.CalledProcessError as e:
        print(f"Error running tool2: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    args = parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output, exist_ok=True)
    
    # tool1を実行
    tool1_output_dir = run_tool1(args)
    
    # tool2を実行
    tool2_output_dir = run_tool2(args, tool1_output_dir)
    
    print("Experiment completed successfully!")
    print(f"tool1 results: {tool1_output_dir}")
    print(f"tool2 results: {tool2_output_dir}")

if __name__ == "__main__":
    main()
