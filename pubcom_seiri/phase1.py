"""
phase1.py: CSVデータの読み込みと重複除去

機能:
- CSVデータの改行は空白を入れずに結合
- 完全一致の重複を検出して除去
- 文字ベースの類似度計算で50%以上共通するコメントをグループ化
- 処理結果をCSVとして出力
"""

import argparse
import csv
import difflib
import json
import os
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='CSVデータの読み込みと重複除去')
    parser.add_argument('--input', required=True, help='入力CSVファイルのパス')
    parser.add_argument('--output', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--limit', type=int, default=10000, help='分析対象とするCSVデータの末尾n件 (default: 10000)')
    parser.add_argument('--similarity', type=float, default=50.0, help='文字ベースの類似度閾値（%） (default: 50.0)')
    return parser.parse_args()

def load_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """CSVからデータを単純に読み込む"""
    print(f"Loading data from {csv_path}...")
    comments = []
    ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader)  # ヘッダーをスキップ
        
        rows = list(reader)
        
        for i, row in enumerate(rows):
            if len(row) >= 1:  # コメント列があることを確認
                comment = "".join(row[0].splitlines())
                id_val = int(row[1]) if len(row) >= 2 and row[1].isdigit() else i
                comments.append(comment)
                ids.append(id_val)
    
    print(f"Loaded {len(comments)} comments.")
    return comments, ids

def remove_duplicates(comments: List[str], ids: List[int]) -> Tuple[List[str], List[int], Dict[str, List[int]]]:
    """重複を検出して除去する"""
    print("Removing duplicates...")
    unique_comments = []
    unique_ids = []
    duplicate_map = {}  # コメント内容をキーに、IDのリストを値とする辞書
    
    for i, comment in enumerate(comments):
        if comment in duplicate_map:
            duplicate_map[comment].append(ids[i])
        else:
            duplicate_map[comment] = [ids[i]]
            unique_comments.append(comment)
            unique_ids.append(ids[i])
    
    duplicates = {comment: id_list for comment, id_list in duplicate_map.items() if len(id_list) > 1}
    
    print(f"Found {len(duplicates)} duplicate comment types.")
    print(f"Reduced to {len(unique_comments)} unique comments.")
    return unique_comments, unique_ids, duplicates

def get_limited_data(comments: List[str], ids: List[int], limit: int = 10000) -> Tuple[List[str], List[int]]:
    """除去済みのデータからN件を取得する"""
    print(f"Getting last {limit} comments...")
    if limit < len(comments):
        limited_comments = comments[-limit:]
        limited_ids = ids[-limit:]
    else:
        limited_comments = comments
        limited_ids = ids
    
    print(f"Selected {len(limited_comments)} comments for analysis.")
    return limited_comments, limited_ids

def calculate_similarity_percentage(text1: str, text2: str) -> float:
    """2つのテキスト間の類似度をパーセンテージで計算する"""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio() * 100

def group_by_similarity(comments: List[str], ids: List[int], similarity_threshold: float = 50.0) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """文字ベースの類似度に基づいてコメントをグループ化する"""
    print(f"Grouping comments by similarity (threshold: {similarity_threshold}%)...")
    
    comment_groups = []
    groups = {}
    processed = set()
    
    for i in range(len(comments)):
        if i in processed:
            continue
        
        group_idx = len(groups)
        groups[group_idx] = [i]
        comment_groups.append(group_idx)
        processed.add(i)
        
        for j in range(i + 1, len(comments)):
            if j in processed:
                continue
            
            similarity = calculate_similarity_percentage(comments[i], comments[j])
            if similarity >= similarity_threshold:
                groups[group_idx].append(j)
                comment_groups.append(group_idx)
                processed.add(j)
    
    id_groups = {group_idx: [ids[comment_idx] for comment_idx in comment_indices] 
                for group_idx, comment_indices in groups.items()}
    
    print(f"Created {len(groups)} similarity-based groups.")
    return list(groups.values()), id_groups

def generate_similarity_report(comments: List[str], groups: List[List[int]], output_dir: str) -> None:
    """類似度グループのレポートを生成する"""
    print("Generating similarity report...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/similarity_groups.md", 'w', encoding='utf-8') as f:
        f.write("# 文字ベースの類似度によるグループ化結果\n\n")
        
        for i, group in enumerate(groups):
            if len(group) > 1:  # 2つ以上のコメントを含むグループのみ表示
                f.write(f"## グループ {i+1} ({len(group)}件)\n\n")
                
                representative_idx = group[0]
                f.write(f"### 代表コメント (インデックス: {representative_idx})\n\n")
                f.write(f"{comments[representative_idx]}\n\n")
                
                f.write("### 類似コメント\n\n")
                for j in group[1:]:
                    similarity = calculate_similarity_percentage(comments[representative_idx], comments[j])
                    f.write(f"- インデックス {j} (類似度: {similarity:.2f}%)\n")
                    f.write(f"  {comments[j]}\n\n")
                
                f.write("---\n\n")
    
    print(f"Similarity report saved to {output_dir}/similarity_groups.md")

def save_results_to_csv(comments: List[str], ids: List[int], groups: List[List[int]], 
                       id_groups: Dict[int, List[int]], duplicates: Dict[str, List[int]], 
                       output_dir: str) -> None:
    """処理結果をCSVとして保存する"""
    print("Saving results to CSV...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    data = []
    for i, comment in enumerate(comments):
        group_idx = None
        for g_idx, group in enumerate(groups):
            if i in group:
                group_idx = g_idx
                break
        
        duplicate_ids = []
        for dup_comment, dup_ids in duplicates.items():
            if comment == dup_comment:
                duplicate_ids = dup_ids
                break
        
        data.append({
            'comment': comment,
            'id': ids[i],
            'group_id': group_idx,
            'group_size': len(groups[group_idx]) if group_idx is not None else 1,
            'duplicate_count': len(duplicate_ids),
            'duplicate_ids': ','.join(map(str, duplicate_ids)) if duplicate_ids else ''
        })
    
    df = pd.DataFrame(data)
    df.to_csv(f"{output_dir}/phase1_results.csv", index=False, encoding='utf-8')
    
    results = {
        "comments": comments,
        "ids": ids,
        "similarity_groups": [list(map(int, group)) for group in groups],
        "id_groups": {str(k): v for k, v in id_groups.items()},
        "duplicates": duplicates
    }
    
    with open(f"{output_dir}/phase1_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_dir}/phase1_results.csv and {output_dir}/phase1_results.json")

def main():
    args = parse_args()
    
    all_comments, all_ids = load_data(args.input)
    
    unique_comments, unique_ids, duplicates = remove_duplicates(all_comments, all_ids)
    
    comments, ids = get_limited_data(unique_comments, unique_ids, args.limit)
    
    groups, id_groups = group_by_similarity(comments, ids, args.similarity)
    
    generate_similarity_report(comments, groups, args.output)
    
    save_results_to_csv(comments, ids, groups, id_groups, duplicates, args.output)
    
    print("Phase 1 processing completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
