#!/usr/bin/env python3
"""
tool1: CSVデータのクラスタリングと結果のHTML出力

機能:
- CSVデータの改行は空白を入れずに結合
- CSVデータの末尾n件を分析対象（デフォルト10000件）
- クラスタリング結果のファイル出力
- 結果を見やすくHTML形式で表示
  - 2つの文字単位diffを取り可視化、共通部の多さをパーセント表示
  - クラスタ内の最も遠いデータとの比較を表示
  - 各データには元CSVでのIDを併記
"""

import argparse
import csv
import difflib
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from typing import List, Dict, Tuple, Any, Optional
import jinja2

def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(description='CSVデータのクラスタリングと結果のHTML出力')
    parser.add_argument('--input', required=True, help='入力CSVファイルのパス')
    parser.add_argument('--output', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--limit', type=int, default=10000, help='分析対象とするCSVデータの末尾n件 (default: 10000)')
    parser.add_argument('--threshold', type=float, default=0.4, help='クラスタリングの距離閾値 (default: 0.4)')
    parser.add_argument('--model', default='sentence-transformers/paraphrase-multilingual-mpnet-base-v2', help='使用する埋め込みモデル名')
    return parser.parse_args()

def load_data(csv_path: str, limit: int = 10000) -> Tuple[List[str], List[int]]:
    """CSVからデータを読み込む"""
    print(f"Loading data from {csv_path}...")
    comments = []
    ids = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # ヘッダーをスキップ
        
        rows = list(reader)
        # 末尾n件を取得
        target_rows = rows[-limit:] if limit < len(rows) else rows
        
        for i, row in enumerate(target_rows):
            if len(row) >= 1:
                # 改行を空白を入れずに結合
                comment = "".join(row[0].splitlines())
                comments.append(comment)
                ids.append(i)
    
    print(f"Loaded {len(comments)} comments.")
    return comments, ids

def create_embeddings(comments: List[str], model_name: str) -> np.ndarray:
    """コメントの埋め込みベクトルを生成する"""
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Creating embeddings...")
    embeddings = model.encode(comments)
    
    return embeddings

def perform_clustering(embeddings: np.ndarray, distance_threshold: float = 0.4) -> Tuple[np.ndarray, Dict[str, List[int]], np.ndarray, np.ndarray]:
    """凝集クラスタリングを実行する"""
    print(f"Performing agglomerative clustering with distance_threshold={distance_threshold}...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='complete'
    )
    
    labels = clustering.fit_predict(embeddings)
    print(f"Found {len(set(labels))} clusters.")
    
    # クラスタごとにインデックスをグループ化
    clusters = {}
    for i, label in enumerate(labels):
        cluster_id = str(label)
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    distances = clustering.distances_
    children = clustering.children_
    
    return labels, clusters, distances, children

def calculate_similarity_percentage(text1: str, text2: str) -> float:
    """2つのテキスト間の類似度をパーセンテージで計算する"""
    matcher = difflib.SequenceMatcher(None, text1, text2)
    return matcher.ratio() * 100

def generate_html_diff(text1: str, text2: str) -> str:
    """2つのテキスト間の差分をHTML形式で生成する"""
    d = difflib.Differ()
    diff = list(d.compare(text1, text2))
    
    html = []
    current_style = None
    current_content = []
    
    for line in diff:
        if line.startswith('  '):
            style = "common"
        elif line.startswith('- '):
            style = "removed"
        elif line.startswith('+ '):
            style = "added"
        else:
            continue
        
        content = line[2:]
        
        if style == current_style:
            current_content.append(content)
        else:
            if current_style is not None:
                html.append(f'<span class="{current_style}">{"".join(current_content)}</span>')
            current_style = style
            current_content = [content]
    
    if current_style is not None:
        html.append(f'<span class="{current_style}">{"".join(current_content)}</span>')
    
    return ''.join(html)

def find_farthest_pair(cluster_indices: List[int], embeddings: np.ndarray) -> Tuple[int, int, float]:
    """クラスタ内で最も遠いペアを見つける"""
    max_distance = -1.0  # floatに明示的に型指定
    farthest_pair = (-1, -1)
    
    for i in range(len(cluster_indices)):
        for j in range(i+1, len(cluster_indices)):
            idx1 = cluster_indices[i]
            idx2 = cluster_indices[j]
            dist = cosine(embeddings[idx1], embeddings[idx2])
            if dist > max_distance:
                max_distance = dist
                farthest_pair = (idx1, idx2)
    
    return farthest_pair[0], farthest_pair[1], max_distance

def extract_merge_info(children: np.ndarray, distances: np.ndarray, comments: List[str], max_merges: int = 50) -> List[Dict[str, Any]]:
    """クラスタ併合情報を抽出する"""
    print(f"近い順に併合される最初の{max_merges}件のクラスタ情報を抽出中...")
    merges = []
    
    sorted_indices = np.argsort(distances)
    sorted_children = children[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    for i in range(min(max_merges, len(sorted_distances))):
        child1, child2 = sorted_children[i]
        distance = sorted_distances[i]
        
        if child1 < len(comments):
            text1 = comments[child1]
            id1 = child1
        else:
            text1 = f"Cluster #{child1 - len(comments)}"
            id1 = child1
            
        if child2 < len(comments):
            text2 = comments[child2]
            id2 = child2
        else:
            text2 = f"Cluster #{child2 - len(comments)}"
            id2 = child2
            
        merges.append({
            'index': i,
            'id1': id1,
            'id2': id2,
            'text1': text1,
            'text2': text2,
            'distance': distance
        })
    
    return merges

def save_merge_info(merges: List[Dict[str, Any]], comments: List[str], output_dir: str) -> None:
    """クラスタ併合情報をMarkdownファイルとして保存する"""
    print("クラスタ併合情報をMarkdownファイルとして保存中...")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/cluster_merges.md", 'w', encoding='utf-8') as f:
        f.write(f"# クラスタ併合情報（最初の{len(merges)}件）\n\n")
        for merge in merges:
            new_cluster_id = len(comments) + merge['index']
            f.write(f"## 併合 #{merge['index'] + 1} （距離: {merge['distance']:.6f}）\n\n")
            f.write(f"**作成されたクラスタID**: {new_cluster_id}\n\n")
            
            if merge['id1'] < len(comments):
                f.write(f"### テキスト1 (ID: {merge['id1']})\n\n")
                f.write(f"{merge['text1']}\n\n")
            else:
                f.write(f"### クラスタ1 (ID: {merge['id1']})\n\n")
                f.write(f"{merge['text1']}\n\n")
            
            if merge['id2'] < len(comments):
                f.write(f"### テキスト2 (ID: {merge['id2']})\n\n")
                f.write(f"{merge['text2']}\n\n")
            else:
                f.write(f"### クラスタ2 (ID: {merge['id2']})\n\n")
                f.write(f"{merge['text2']}\n\n")
            
            if merge['id1'] < len(comments) and merge['id2'] < len(comments):
                similarity = calculate_similarity_percentage(merge['text1'], merge['text2'])
                diff = generate_html_diff(merge['text1'], merge['text2'])
                f.write(f"### 類似度: {similarity:.2f}%\n\n")
                f.write(f"### 差分\n\n```html\n{diff}\n```\n\n")
            
            f.write("---\n\n")
        
        import pandas as pd
        df = pd.DataFrame(merges)
        df.to_csv(f"{output_dir}/cluster_merges.csv", index=False, encoding='utf-8')

def generate_html_report(clusters: Dict[str, List[int]], comments: List[str], ids: List[int], 
                        embeddings: np.ndarray, output_dir: str) -> None:
    """HTML形式のレポートを生成する"""
    print("Generating HTML report...")
    
    # Jinja2テンプレート環境を設定
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>クラスタリング結果</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #555; margin-top: 30px; }
            .cluster { margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
            .comment { margin-bottom: 20px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
            .comment-id { font-weight: bold; margin-bottom: 5px; }
            .comment-text { white-space: pre-wrap; }
            .diff-container { margin-top: 15px; border-top: 1px dashed #ccc; padding-top: 15px; }
            .diff-title { font-weight: bold; margin-bottom: 10px; }
            .diff { white-space: pre-wrap; font-family: monospace; }
            .common { color: black; }
            .removed { color: red; text-decoration: line-through; }
            .added { color: green; }
            .similarity { font-weight: bold; color: #0066cc; }
            .summary { margin-bottom: 20px; padding: 10px; background-color: #eef; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>クラスタリング結果</h1>
        
        <div class="summary">
            <p>総クラスタ数: {{ clusters|length }}</p>
            <p>総コメント数: {{ total_comments }}</p>
            <p>クラスタサイズ分布:</p>
            <ul>
                {% for size, count in cluster_sizes.items() %}
                <li>{{ size }}件のクラスタ: {{ count }}個</li>
                {% endfor %}
            </ul>
        </div>
        
        {% for cluster_id, cluster_data in cluster_results.items() %}
        <div class="cluster">
            <h2>クラスタ #{{ cluster_id }} ({{ cluster_data.indices|length }}件)</h2>
            
            {% if cluster_data.indices|length >= 2 %}
            <div class="diff-container">
                <div class="diff-title">クラスタ内の最も遠いペア (距離: {{ "%.4f"|format(cluster_data.max_distance) }}, 類似度: {{ "%.2f"|format(cluster_data.similarity) }}%)</div>
                <div class="diff">{{ cluster_data.diff|safe }}</div>
            </div>
            {% endif %}
            
            {% for idx in cluster_data.indices %}
            <div class="comment">
                <div class="comment-id">ID: {{ ids[idx] }}</div>
                <div class="comment-text">{{ comments[idx] }}</div>
            </div>
            {% endfor %}
        </div>
        {% endfor %}
    </body>
    </html>
    """
    
    template = jinja2.Template(template_str)
    
    # クラスタサイズの分布を計算
    cluster_sizes = {}
    for indices in clusters.values():
        size = len(indices)
        if size not in cluster_sizes:
            cluster_sizes[size] = 0
        cluster_sizes[size] += 1
    
    # クラスタごとの結果を準備
    cluster_results = {}
    for cluster_id, indices in clusters.items():
        if len(indices) >= 2:
            idx1, idx2, max_distance = find_farthest_pair(indices, embeddings)
            similarity = calculate_similarity_percentage(comments[idx1], comments[idx2])
            diff = generate_html_diff(comments[idx1], comments[idx2])
        else:
            max_distance = 0
            similarity = 100
            diff = ""
        
        cluster_results[cluster_id] = {
            "indices": indices,
            "max_distance": max_distance,
            "similarity": similarity,
            "diff": diff
        }
    
    # HTMLを生成
    html = template.render(
        clusters=clusters,
        total_comments=len(comments),
        cluster_sizes=cluster_sizes,
        cluster_results=cluster_results,
        comments=comments,
        ids=ids
    )
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # HTMLファイルを保存
    with open(f"{output_dir}/clustering_report.html", 'w', encoding='utf-8') as f:
        f.write(html)
    
    # クラスタリング結果をJSONとしても保存
    results = {
        "comments": comments,
        "ids": ids,
        "labels": [int(label) for label in list(map(str, clusters.keys()))],
        "clusters": clusters
    }
    
    with open(f"{output_dir}/clusters.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    
    # データを読み込む
    comments, ids = load_data(args.input, args.limit)
    
    # 埋め込みベクトルを生成
    embeddings = create_embeddings(comments, args.model)
    
    # クラスタリングを実行
    labels, clusters, distances, children = perform_clustering(embeddings, args.threshold)
    
    merges = extract_merge_info(children, distances, comments, max_merges=50)
    
    save_merge_info(merges, comments, args.output)
    
    # HTMLレポートを生成
    generate_html_report(clusters, comments, ids, embeddings, args.output)
    
    print("Processing completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
