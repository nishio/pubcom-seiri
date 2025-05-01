#!/usr/bin/env python3
"""
tool2: クラスタリング結果の詳細分析と出力

機能:
- tool1を見ることでどこまでを類似とするかの閾値が決まる、これを引数として取る
- clusteringの結果、2件以上のクラスタになったものとそれ以外(独立点)に分ける
- クラスタの中央を計算しておく
- "csv dataの末尾n件を分析対象にする"ことで除外されていたデータが存在するなら振り分け作業をする
  - 各クラスタ中央とデータの距離を計算し、それが閾値を超えているなら最遠点も超えているので対象外、超えてないクラスタに対して最遠点との距離を計算し最も近いところに入れる。対象となるクラスタが存在しないならクラスタに入れないで「独立点」となる
- 出力データは、独立点のリスト、各クラスタごとのリスト(一行目をクラスタ中央に最も近いデータ="代表点"にする) 
- 後段の高度な分析のために独立点のリストと各クラスタの代表点のリストを結合したものも作る
- どのデータにおいても入力データにおけるIDを併記すること
- データ分析の各種パラメータや、その結果クラスタがいくつ作られて、何件がクラスタに分類されたか、それは全体の何%であるか、最も大きなクラスタは何か、などの人間が読むためのMarkdownレポートを生成せよ
"""

import argparse
import csv
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='クラスタリング結果の詳細分析と出力')
    parser.add_argument('--input', required=True, help='tool1の出力ディレクトリのパス（clusters.jsonがあるディレクトリ）')
    parser.add_argument('--original', help='元のCSVファイルのパス（除外されていたデータを処理する場合）')
    parser.add_argument('--threshold', type=float, required=True, help='類似度の閾値')
    parser.add_argument('--output', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='使用する埋め込みモデル名')
    return parser.parse_args()

def load_clustering_results(input_dir: str) -> Dict[str, Any]:
    """tool1の出力結果を読み込む"""
    print(f"Loading clustering results from {input_dir}...")
    with open(f"{input_dir}/clusters.json", 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def load_excluded_data(csv_path: str, included_ids: List[int]) -> Tuple[List[str], List[int]]:
    """除外されていたデータを読み込む"""
    print(f"Loading excluded data from {csv_path}...")
    excluded_comments = []
    excluded_ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダーをスキップ
        
        for i, row in enumerate(reader):
            if i not in included_ids and len(row) >= 1:
                # 改行を空白を入れずに結合
                comment = "".join(row[0].splitlines())
                excluded_comments.append(comment)
                excluded_ids.append(i)
    
    print(f"Loaded {len(excluded_comments)} excluded comments.")
    return excluded_comments, excluded_ids

def calculate_cluster_centers(clusters: Dict[str, List[int]], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    """各クラスタの中心を計算する"""
    print("Calculating cluster centers...")
    centers = {}
    
    for cluster_id, indices in clusters.items():
        if len(indices) >= 2:  # 2件以上のクラスタのみ処理
            cluster_embeddings = embeddings[indices]
            center = np.mean(cluster_embeddings, axis=0)
            centers[cluster_id] = center
    
    return centers

def find_closest_to_center(cluster_indices: List[int], center: np.ndarray, embeddings: np.ndarray) -> int:
    """クラスタ中心に最も近いデータのインデックスを見つける"""
    min_distance = float('inf')
    closest_idx = -1
    
    for idx in cluster_indices:
        if idx < len(embeddings):
            dist = cosine(embeddings[idx], center)
            if dist < min_distance:
                min_distance = dist
                closest_idx = idx
    
    return closest_idx

def find_farthest_from_center(cluster_indices: List[int], center: np.ndarray, embeddings: np.ndarray) -> Tuple[int, float]:
    """クラスタ中心から最も遠いデータのインデックスと距離を見つける"""
    max_distance = -1
    farthest_idx = -1
    
    for idx in cluster_indices:
        if idx < len(embeddings):
            dist = cosine(embeddings[idx], center)
            if dist > max_distance:
                max_distance = dist
                farthest_idx = idx
    
    return farthest_idx, max_distance

def assign_excluded_data(excluded_comments: List[str], excluded_ids: List[int], 
                        clusters: Dict[str, List[int]], centers: Dict[str, np.ndarray], 
                        embeddings: np.ndarray, threshold: float, model_name: str) -> Tuple[Dict[str, List[int]], List[int]]:
    """除外されていたデータをクラスタに振り分ける"""
    if not excluded_comments:
        return clusters, []
    
    print(f"Assigning {len(excluded_comments)} excluded comments to clusters...")
    
    # 埋め込みベクトルを生成
    model = SentenceTransformer(model_name)
    
    # バッチサイズを設定して処理
    batch_size = 100
    excluded_embeddings = []
    
    for i in range(0, len(excluded_comments), batch_size):
        batch = excluded_comments[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        excluded_embeddings.append(batch_embeddings)
    
    # バッチの結果を結合
    excluded_embeddings = np.vstack(excluded_embeddings) if len(excluded_embeddings) > 0 else np.array([])
    
    # 独立点のリスト
    independent_points = []
    
    # 新しいクラスタ辞書を作成
    new_clusters = {k: v.copy() for k, v in clusters.items()}
    
    # 各除外データに対して
    for i, (comment, comment_id) in enumerate(zip(excluded_comments, excluded_ids)):
        embedding = excluded_embeddings[i]
        
        # 各クラスタ中心との距離を計算
        min_distance = float('inf')
        closest_cluster = None
        
        for cluster_id, center in centers.items():
            dist = cosine(embedding, center)
            if dist < min_distance:
                min_distance = dist
                closest_cluster = cluster_id
        
        # 最も近いクラスタが閾値内にあるか確認
        if closest_cluster and min_distance <= threshold:
            # クラスタ内の最も遠い点との距離を計算
            cluster_indices = clusters[closest_cluster]
            farthest_idx, farthest_dist = find_farthest_from_center(cluster_indices, centers[closest_cluster], embeddings)
            
            # 最も遠い点よりも近ければクラスタに追加
            if min_distance <= farthest_dist:
                # 元のインデックスリストの長さを取得
                original_len = len(embeddings)
                
                # 新しいインデックスを計算（元のembeddingsの長さ + 現在の位置）
                new_idx = original_len + i
                
                # クラスタに追加
                new_clusters[closest_cluster].append(new_idx)
            else:
                # 独立点として追加
                independent_points.append(comment_id)
        else:
            # 独立点として追加
            independent_points.append(comment_id)
    
    return new_clusters, independent_points

def generate_output_data(clusters: Dict[str, List[int]], centers: Dict[str, np.ndarray], 
                        embeddings: np.ndarray, comments: List[str], ids: List[int],
                        independent_points: List[int], excluded_comments: Optional[List[str]] = None,
                        excluded_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """出力データを生成する"""
    print("Generating output data...")
    
    # 独立点のリスト
    independent_list = []
    for idx in independent_points:
        if idx < len(comments):
            # 元のデータからの独立点
            independent_list.append({
                "comment": comments[idx],
                "id": ids[idx]
            })
        elif excluded_comments and excluded_ids:
            # 除外データからの独立点
            excluded_idx = idx - len(comments)
            if excluded_idx < len(excluded_comments):
                independent_list.append({
                    "comment": excluded_comments[excluded_idx],
                    "id": excluded_ids[excluded_idx]
                })
    
    # クラスタごとのリスト
    cluster_lists = {}
    representative_points = []
    
    for cluster_id, indices in clusters.items():
        if len(indices) >= 2:  # 2件以上のクラスタのみ処理
            center = centers[cluster_id]
            
            # クラスタ中心に最も近いデータを見つける（代表点）
            closest_idx = find_closest_to_center(indices, center, embeddings)
            
            # クラスタリスト
            cluster_list = []
            
            # 代表点を最初に追加
            if closest_idx < len(comments):
                rep_point = {
                    "comment": comments[closest_idx],
                    "id": ids[closest_idx]
                }
                cluster_list.append(rep_point)
                representative_points.append(rep_point)
            
            # 残りのポイントを追加
            for idx in indices:
                if idx != closest_idx and idx < len(comments):
                    cluster_list.append({
                        "comment": comments[idx],
                        "id": ids[idx]
                    })
            
            cluster_lists[cluster_id] = cluster_list
    
    # 独立点と代表点を結合したリスト
    combined_list = independent_list + representative_points
    
    return {
        "independent_points": independent_list,
        "cluster_lists": cluster_lists,
        "combined_list": combined_list
    }

def save_output_data(output_data: Dict[str, Any], output_dir: str) -> None:
    """出力データを保存する"""
    print(f"Saving output data to {output_dir}...")
    
    # JSONとして保存
    with open(f"{output_dir}/independent_points.json", 'w', encoding='utf-8') as f:
        json.dump(output_data["independent_points"], f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/cluster_lists.json", 'w', encoding='utf-8') as f:
        json.dump(output_data["cluster_lists"], f, ensure_ascii=False, indent=2)
    
    with open(f"{output_dir}/combined_list.json", 'w', encoding='utf-8') as f:
        json.dump(output_data["combined_list"], f, ensure_ascii=False, indent=2)
    
    # CSVとして保存
    # 独立点のCSV
    with open(f"{output_dir}/independent_points.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment"])
        for point in output_data["independent_points"]:
            writer.writerow([point["id"], point["comment"]])
    
    # クラスタごとのCSV
    for cluster_id, cluster_list in output_data["cluster_lists"].items():
        with open(f"{output_dir}/cluster_{cluster_id}.csv", 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "comment", "is_representative"])
            for i, point in enumerate(cluster_list):
                writer.writerow([point["id"], point["comment"], i == 0])
    
    # 結合リストのCSV
    with open(f"{output_dir}/combined_list.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment", "type"])
        for point in output_data["independent_points"]:
            writer.writerow([point["id"], point["comment"], "independent"])
        for point in output_data["combined_list"][len(output_data["independent_points"]):]:
            writer.writerow([point["id"], point["comment"], "representative"])

def generate_markdown_report(output_data: Dict[str, Any], clusters: Dict[str, List[int]], 
                           independent_points: List[int], threshold: float, 
                           total_comments: int, output_dir: str,
                           duplicates: Dict[str, List[int]] = None, ids: List[int] = None) -> None:
    """Markdownレポートを生成する"""
    print("Generating Markdown report...")
    
    # クラスタ数
    num_clusters = len([k for k, v in clusters.items() if len(v) >= 2])
    
    # クラスタに分類されたコメント数
    clustered_comments = sum(len(v) for k, v in clusters.items() if len(v) >= 2)
    
    # 独立点の数
    num_independent = len(independent_points)
    
    # 最大のクラスタ
    max_cluster_id = None
    max_cluster_size = 0
    for cluster_id, indices in clusters.items():
        if len(indices) >= 2 and len(indices) > max_cluster_size:
            max_cluster_size = len(indices)
            max_cluster_id = cluster_id
    
    duplicate_section = ""
    if duplicates and len(duplicates) > 0:
        duplicate_section = "\n## 完全一致のコメント\n"
        for comment, indices in duplicates.items():
            if len(indices) > 1:
                id_str = ", ".join([str(ids[idx]) for idx in indices])
                duplicate_section += f"- ID {id_str}は同一内容が{len(indices)}件あった\n"
    
    # レポート作成
    report = f"""# クラスタリング分析レポート

## 分析パラメータ
- 類似度閾値: {threshold}
- 総コメント数: {total_comments}

## 分析結果
- クラスタ数: {num_clusters}
- クラスタに分類されたコメント数: {clustered_comments} ({clustered_comments/total_comments*100:.2f}%)
- 独立点の数: {num_independent} ({num_independent/total_comments*100:.2f}%)
{duplicate_section}
## 最大のクラスタ
- クラスタID: {max_cluster_id}
- サイズ: {max_cluster_size} コメント ({max_cluster_size/total_comments*100:.2f}%)

## 出力ファイル
- `independent_points.json`: 独立点のリスト（JSON形式）
- `independent_points.csv`: 独立点のリスト（CSV形式）
- `cluster_lists.json`: クラスタごとのリスト（JSON形式）
- `combined_list.json`: 独立点と代表点を結合したリスト（JSON形式）
- `combined_list.csv`: 独立点と代表点を結合したリスト（CSV形式）
- `cluster_*.csv`: 各クラスタのリスト（CSV形式）
"""
    
    # レポートを保存
    with open(f"{output_dir}/report.md", 'w', encoding='utf-8') as f:
        f.write(report)

def main():
    """メイン関数"""
    args = parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output, exist_ok=True)
    
    # クラスタリング結果を読み込む
    results = load_clustering_results(args.input)
    comments = results['comments']
    ids = results['ids']
    clusters = results['clusters']
    duplicates = results.get('duplicates', {})
    
    # 埋め込みベクトルを再作成
    print("Recreating embeddings...")
    model = SentenceTransformer(args.model)
    embeddings = model.encode(comments)
    
    # クラスタ中心を計算
    centers = calculate_cluster_centers(clusters, embeddings)
    
    # 独立点を抽出
    independent_points = []
    for i in range(len(comments)):
        is_in_cluster = False
        for cluster_id, indices in clusters.items():
            if i in indices and len(indices) >= 2:
                is_in_cluster = True
                break
        if not is_in_cluster:
            independent_points.append(i)
    
    # 除外データを処理
    excluded_comments = None
    excluded_ids = None
    if args.original:
        excluded_comments, excluded_ids = load_excluded_data(args.original, ids)
        if excluded_comments:
            clusters, additional_independent_points = assign_excluded_data(
                excluded_comments, excluded_ids, clusters, centers, 
                embeddings, args.threshold, args.model
            )
            independent_points.extend(additional_independent_points)
    
    # 出力データを生成
    output_data = generate_output_data(
        clusters, centers, embeddings, comments, ids,
        independent_points, excluded_comments, excluded_ids
    )
    
    # 出力データを保存
    save_output_data(output_data, args.output)
    
    # Markdownレポートを生成
    total_comments = len(comments)
    if excluded_comments:
        total_comments += len(excluded_comments)
    
    generate_markdown_report(
        output_data, clusters, independent_points,
        args.threshold, total_comments, args.output,
        duplicates, ids
    )
    
    print(f"Processing completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
