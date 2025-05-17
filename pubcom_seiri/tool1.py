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


def select_column(row):
    "return (id, comment)"
    # aipubcom
    # comment = row[0]
    # id_val = int(row[1])

    # ene
    comment = row[2]
    id_val = int(row[0])

    return (id_val, comment)


def parse_args():
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description="CSVデータのクラスタリングと結果のHTML出力"
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
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="使用する埋め込みモデル名",
    )
    return parser.parse_args()


def load_data(csv_path: str) -> Tuple[List[str], List[int]]:
    """CSVからデータを単純に読み込む"""
    print(f"Loading data from {csv_path}...")
    comments = []
    ids = []

    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)  # ヘッダーをスキップ

        rows = list(reader)

        for i, row in enumerate(rows):
            if len(row) >= 1:  # コメント列があることを確認
                id_val, comment = select_column(row)
                # 改行を空白を入れずに結合
                comment = "".join(comment.splitlines())
                comments.append(comment)
                ids.append(id_val)

    print(f"Loaded {len(comments)} comments.")
    return comments, ids


def remove_duplicates(
    comments: List[str], ids: List[int]
) -> Tuple[List[str], List[int], Dict[str, List[int]], Dict[int, List[int]]]:
    """重複を検出して除去する"""
    print("Removing duplicates...")
    unique_comments = []
    unique_ids = []
    duplicate_map = {}  # コメント内容をキーに、IDのリストを値とする辞書
    id_mapping = {}     # インデックスをキーに、元のCSV IDのリストを値とする辞書

    for i, comment in enumerate(comments):
        if comment in duplicate_map:
            duplicate_map[comment].append(ids[i])
        else:
            duplicate_map[comment] = [ids[i]]
            unique_comments.append(comment)
            unique_ids.append(ids[i])
            id_mapping[len(unique_comments) - 1] = [ids[i]]  # 新しいインデックスに対応するIDを記録

    for i, comment in enumerate(unique_comments):
        id_mapping[i] = duplicate_map[comment]  # 元のCSV IDのリストを保存

    duplicates = {
        comment: id_list
        for comment, id_list in duplicate_map.items()
        if len(id_list) > 1
    }

    print(f"Found {len(duplicates)} duplicate comment types.")
    print(f"Reduced to {len(unique_comments)} unique comments.")
    return unique_comments, unique_ids, duplicates, id_mapping


def get_limited_data(
    comments: List[str], ids: List[int], id_mapping: Dict[int, List[int]] = None, limit: int = 10000
) -> Tuple[List[str], List[int], Dict[int, List[int]]]:
    """除去済みのデータからN件を取得する"""
    print(f"Getting last {limit} comments...")
    if limit < len(comments):
        limited_comments = comments[-limit:]
        limited_ids = ids[-limit:]
        if id_mapping:
            limited_id_mapping = {}
            for i in range(len(limited_comments)):
                original_idx = len(comments) - limit + i
                limited_id_mapping[i] = id_mapping[original_idx]
        else:
            limited_id_mapping = None
    else:
        limited_comments = comments
        limited_ids = ids
        limited_id_mapping = id_mapping

    print(f"Selected {len(limited_comments)} comments for analysis.")
    return limited_comments, limited_ids, limited_id_mapping



def create_embeddings(comments: List[str], model_name: str) -> np.ndarray:
    """コメントの埋め込みベクトルを生成する"""
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print("Creating embeddings...")
    embeddings = model.encode(comments)

    return embeddings


def perform_clustering(
    embeddings: np.ndarray, distance_threshold: float = 0.4
) -> Tuple[np.ndarray, Dict[str, List[int]], np.ndarray, np.ndarray]:
    """凝集クラスタリングを実行する"""
    print(
        f"Performing agglomerative clustering with distance_threshold={distance_threshold}..."
    )
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="complete",
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

    distances = np.array(clustering.distances_)
    children = np.array(clustering.children_)

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
        if line.startswith("  "):
            style = "common"
        elif line.startswith("- "):
            style = "removed"
        elif line.startswith("+ "):
            style = "added"
        else:
            continue

        content = line[2:]

        if style == current_style:
            current_content.append(content)
        else:
            if current_style is not None:
                html.append(
                    f'<span class="{current_style}">{"".join(current_content)}</span>'
                )
            current_style = style
            current_content = [content]

    if current_style is not None:
        html.append(f'<span class="{current_style}">{"".join(current_content)}</span>')

    return "".join(html)


def find_farthest_pair(
    cluster_indices: List[int], embeddings: np.ndarray
) -> Tuple[int, int, float]:
    """クラスタ内で最も遠いペアを見つける"""
    max_distance = -1.0  # floatに明示的に型指定
    farthest_pair = (-1, -1)

    for i in range(len(cluster_indices)):
        for j in range(i + 1, len(cluster_indices)):
            idx1 = cluster_indices[i]
            idx2 = cluster_indices[j]
            dist = cosine(embeddings[idx1], embeddings[idx2])
            if dist > max_distance:
                max_distance = dist
                farthest_pair = (idx1, idx2)

    return farthest_pair[0], farthest_pair[1], float(max_distance)


def extract_merge_info(
    children: np.ndarray,
    distances: np.ndarray,
    comments: List[str],
    embeddings: np.ndarray,
    max_merges: int = -1,
    id_mapping: Dict[int, List[int]] = None,  # id_mappingパラメータを追加
) -> List[Dict[str, Any]]:
    """クラスタ併合情報を抽出する"""
    if max_merges < 0:
        print("全ての併合過程を抽出中...")
        max_items = len(distances)
    else:
        print(f"近い順に併合される最初の{max_merges}件のクラスタ情報を抽出中...")
        max_items = min(max_merges, len(distances))
    merges = []

    cluster_contents = {}
    for i in range(len(comments)):
        cluster_contents[i] = [i]

    sorted_indices = np.argsort(distances)
    sorted_children = children[sorted_indices]
    sorted_distances = distances[sorted_indices]

    for i in range(max_items):
        try:
            child1, child2 = sorted_children[i]
            distance = sorted_distances[i]

            child1 = int(child1)
            child2 = int(child2)

            if child1 < len(comments):
                text1 = comments[child1]
                id1 = child1
                text1_info = None
            else:
                if child1 not in cluster_contents:
                    print(
                        f"警告: クラスタID {child1} が見つかりません。スキップします。"
                    )
                    continue

                cluster_indices = cluster_contents[child1]
                cluster_size = len(cluster_indices)

                if cluster_size >= 2:
                    local_idx1, local_idx2, _ = find_farthest_pair(
                        range(len(cluster_indices)), embeddings[cluster_indices]
                    )
                    representative_idx = cluster_indices[local_idx1]
                    text1 = comments[representative_idx]
                    text1_info = {
                        "text_id": representative_idx,
                        "cluster_id": child1,
                        "cluster_size": cluster_size,
                    }
                else:
                    representative_idx = cluster_indices[0]
                    text1 = comments[representative_idx]
                    text1_info = {
                        "text_id": representative_idx,
                        "cluster_id": child1,
                        "cluster_size": cluster_size,
                    }
                id1 = child1

            if child2 < len(comments):
                text2 = comments[child2]
                id2 = child2
                text2_info = None
            else:
                if child2 not in cluster_contents:
                    print(
                        f"警告: クラスタID {child2} が見つかりません。スキップします。"
                    )
                    continue

                cluster_indices = cluster_contents[child2]
                cluster_size = len(cluster_indices)

                if cluster_size >= 2:
                    local_idx1, local_idx2, _ = find_farthest_pair(
                        range(len(cluster_indices)), embeddings[cluster_indices]
                    )
                    representative_idx = cluster_indices[local_idx1]
                    text2 = comments[representative_idx]
                    text2_info = {
                        "text_id": representative_idx,
                        "cluster_id": child2,
                        "cluster_size": cluster_size,
                    }
                else:
                    representative_idx = cluster_indices[0]
                    text2 = comments[representative_idx]
                    text2_info = {
                        "text_id": representative_idx,
                        "cluster_id": child2,
                        "cluster_size": cluster_size,
                    }
                id2 = child2
        except KeyError as e:
            print(f"警告: 併合 #{i} の処理中にKeyErrorが発生しました: {e}")
            continue
        except Exception as e:
            print(f"警告: 併合 #{i} の処理中にエラーが発生しました: {e}")
            continue

        merges.append(
            {
                "index": i,
                "id1": id1,
                "id2": id2,
                "text1": text1,
                "text2": text2,
                "text1_info": text1_info,
                "text2_info": text2_info,
                "distance": distance,
            }
        )

        new_cluster_id = len(comments) + i
        cluster_contents[new_cluster_id] = (
            cluster_contents[child1] + cluster_contents[child2]
        )

    return merges


def save_merge_info(
    merges: List[Dict[str, Any]], comments: List[str], output_dir: str
) -> None:
    """クラスタ併合情報をMarkdownファイルとして保存する"""
    print("クラスタ併合情報をMarkdownファイルとして保存中...")

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/cluster_merges.md", "w", encoding="utf-8") as f:
        f.write(f"# クラスタ併合情報（最初の{len(merges)}件）\n\n")
        for merge in merges:
            new_cluster_id = len(comments) + merge["index"]
            f.write(
                f"## 併合 #{merge['index'] + 1} （距離: {merge['distance']:.6f}）\n\n"
            )
            f.write(f"**作成されたクラスタID**: {new_cluster_id}\n\n")

            if merge["id1"] < len(comments):
                f.write(f"### テキスト1 (ID: {merge['id1']})\n\n")
                f.write(f"{merge['text1']}\n\n")
            else:
                text1_info = merge["text1_info"]
                f.write(
                    f"### テキスト{text1_info['text_id']}(ID: {text1_info['text_id']}) from クラスタ1 (ID: {merge['id1']}, サイズ{text1_info['cluster_size']})\n\n"
                )
                f.write(f"{merge['text1']}\n\n")

            if merge["id2"] < len(comments):
                f.write(f"### テキスト2 (ID: {merge['id2']})\n\n")
                f.write(f"{merge['text2']}\n\n")
            else:
                text2_info = merge["text2_info"]
                f.write(
                    f"### テキスト{text2_info['text_id']}(ID: {text2_info['text_id']}) from クラスタ2 (ID: {merge['id2']}, サイズ{text2_info['cluster_size']})\n\n"
                )
                f.write(f"{merge['text2']}\n\n")

            if merge["id1"] < len(comments) and merge["id2"] < len(comments):
                similarity = calculate_similarity_percentage(
                    merge["text1"], merge["text2"]
                )
                diff = generate_html_diff(merge["text1"], merge["text2"])
                f.write(f"### 類似度: {similarity:.2f}%\n\n")
                f.write(f"### 差分\n\n```html\n{diff}\n```\n\n")

            f.write("---\n\n")

        import pandas as pd

        df = pd.DataFrame(merges)
        # text1_info と text2_info は複雑なオブジェクトなのでCSVに保存する前に削除
        df_csv = df.copy()
        if "text1_info" in df_csv.columns:
            df_csv = df_csv.drop(columns=["text1_info", "text2_info"])
        df_csv.to_csv(f"{output_dir}/cluster_merges.csv", index=False, encoding="utf-8")


def generate_html_report(
    clusters: Dict[str, List[int]],
    comments: List[str],
    ids: List[int],
    embeddings: np.ndarray,
    merges: List[Dict[str, Any]],
    output_dir: str,
    duplicates: Dict[str, List[int]] = None,
    id_mapping: Dict[int, List[int]] = None,
) -> None:
    """HTML形式のレポートを生成する"""
    print("Generating HTML report...")

    merge_similarities = []
    merge_diffs = []
    for merge in merges:
        similarity = calculate_similarity_percentage(merge["text1"], merge["text2"])
        diff = generate_html_diff(merge["text1"], merge["text2"])
        merge_similarities.append(similarity)
        merge_diffs.append(diff)

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
            h3 { color: #666; }
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
            .merge-process { margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f5f5f5; }
            .merge-item { margin-bottom: 20px; padding: 10px; background-color: #fff; border-radius: 3px; border-left: 4px solid #0066cc; }
            .merge-title { font-weight: bold; margin-bottom: 10px; color: #0066cc; }
            .merge-texts { display: flex; gap: 20px; }
            .merge-text { flex: 1; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
            .merge-diff { margin-top: 15px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
            .duplicates { margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #fff8e1; }
            .duplicate-item { margin-bottom: 10px; padding: 5px; background-color: #fffde7; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>クラスタリング結果</h1>
        
        <div class="summary">
            <p>総コメント数: {{ total_comments }}</p>
        </div>
        
        {% if duplicates and duplicates|length > 0 %}
        <h2>完全一致のコメント</h2>
        <div class="duplicates">
            <p>完全に同一内容のコメント: {{ duplicates|length }}種類</p>
            <ul>
                {% for comment, indices in duplicates.items() %}
                <li class="duplicate-item">
                    同一内容{{ indices|length }}件: ID {{ ids_str[loop.index0] }}
                    <div class="comment-text">{{ comment }}</div>
                </li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <h2>クラスタ併合過程</h2>
        <div class="merge-process">
            <p>閾値に至るまでの併合過程を表示しています（最初の{{ merges|length }}件）</p>
            
            {% for merge in merges %}
            <div class="merge-item">
                <div class="merge-title">併合 #{{ merge.index + 1 }} （距離: {{ "%.6f"|format(merge.distance) }}）</div>
                
                <div class="merge-texts">
                    <div class="merge-text">
                        <h3>{% if merge.id1 < total_comments %}テキスト1 ({% if id_mapping and merge.id1 in id_mapping %}CSV ID: {{ id_mapping[merge.id1]|join(', ') }}{% else %}ID: {{ merge.id1 }}{% endif %}){% else %}テキスト{{ merge.text1_info.text_id }}({% if id_mapping and merge.text1_info.text_id in id_mapping %}CSV ID: {{ id_mapping[merge.text1_info.text_id]|join(', ') }}{% else %}ID: {{ merge.text1_info.text_id }}{% endif %}) from クラスタ1 (ID: {{ merge.id1 }}, サイズ{{ merge.text1_info.cluster_size }}){% endif %}</h3>
                        <div>{{ merge.text1 }}</div>
                    </div>
                    
                    <div class="merge-text">
                        <h3>{% if merge.id2 < total_comments %}テキスト2 ({% if id_mapping and merge.id2 in id_mapping %}CSV ID: {{ id_mapping[merge.id2]|join(', ') }}{% else %}ID: {{ merge.id2 }}{% endif %}){% else %}テキスト{{ merge.text2_info.text_id }}({% if id_mapping and merge.text2_info.text_id in id_mapping %}CSV ID: {{ id_mapping[merge.text2_info.text_id]|join(', ') }}{% else %}ID: {{ merge.text2_info.text_id }}{% endif %}) from クラスタ2 (ID: {{ merge.id2 }}, サイズ{{ merge.text2_info.cluster_size }}){% endif %}</h3>
                        <div>{{ merge.text2 }}</div>
                    </div>
                </div>
                
                <div class="merge-diff">
                    <h3>類似度: {{ "%.2f"|format(merge_similarities[loop.index0]) }}%</h3>
                    <div class="diff">{{ merge_diffs[loop.index0]|safe }}</div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <h2>最終クラスタリング結果</h2>
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
                <div class="comment-id">
                    {% if id_mapping and idx in id_mapping %}
                        CSV ID: {{ id_mapping[idx]|join(', ') }}
                    {% else %}
                        ID: {{ ids[idx] }}
                    {% endif %}
                </div>
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
            "diff": diff,
        }

    ids_str = []
    if duplicates:
        for comment, indices in duplicates.items():
            ids_str.append(", ".join([str(idx) for idx in indices]))

    # HTMLを生成
    html = template.render(
        clusters=clusters,
        total_comments=len(comments),
        cluster_sizes=cluster_sizes,
        cluster_results=cluster_results,
        comments=comments,
        ids=ids,
        merges=merges,
        merge_similarities=merge_similarities,
        merge_diffs=merge_diffs,
        duplicates=duplicates,
        ids_str=ids_str,
        id_mapping=id_mapping,
    )

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # HTMLファイルを保存
    with open(f"{output_dir}/clustering_report.html", "w", encoding="utf-8") as f:
        f.write(html)

    # クラスタリング結果をJSONとしても保存
    results = {
        "comments": comments,
        "ids": ids,
        "labels": [int(label) for label in list(map(str, clusters.keys()))],
        "clusters": clusters,
        "duplicates": duplicates if duplicates else {},
    }

    with open(f"{output_dir}/clusters.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    all_comments, all_ids = load_data(args.input)

    unique_comments, unique_ids, duplicates, id_mapping = remove_duplicates(all_comments, all_ids)

    comments, ids, id_mapping = get_limited_data(unique_comments, unique_ids, id_mapping, args.limit)

    # 埋め込みベクトルを生成
    embeddings = create_embeddings(comments, args.model)

    # クラスタリングを実行
    labels, clusters, distances, children = perform_clustering(
        embeddings, args.threshold
    )

    merges = extract_merge_info(
        children, distances, comments, embeddings, max_merges=1000, id_mapping=id_mapping
    )

    save_merge_info(merges, comments, args.output)

    # HTMLレポートを生成
    generate_html_report(
        clusters, comments, ids, embeddings, merges, args.output, duplicates, id_mapping
    )

    print("Processing completed successfully!")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
