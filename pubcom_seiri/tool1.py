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
    # comment = row[2]
    # id_val = int(row[0])

    # kokkosho
    id_val = int(row[0])  # 受付番号
    comment = row[1]  # 提出意見

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

    # CSVフィールドサイズの制限を増やす
    csv.field_size_limit(1000000)  # 1MB まで許可

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
    id_mapping = {}  # インデックスをキーに、元のCSV IDのリストを値とする辞書

    for i, comment in enumerate(comments):
        if comment in duplicate_map:
            duplicate_map[comment].append(ids[i])
        else:
            duplicate_map[comment] = [ids[i]]
            unique_comments.append(comment)
            unique_ids.append(ids[i])
            id_mapping[len(unique_comments) - 1] = [
                ids[i]
            ]  # 新しいインデックスに対応するIDを記録

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
    comments: List[str],
    ids: List[int],
    id_mapping: Dict[int, List[int]] = None,
    limit: int = 10000,
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

    all_possible_ids = set()
    for child1, child2 in children:
        all_possible_ids.add(int(child1))
        all_possible_ids.add(int(child2))

    max_cluster_id = len(comments) + len(distances)

    for i in range(len(comments), max_cluster_id):
        if i not in cluster_contents:
            cluster_contents[i] = []
    for i in range(max_items):
        try:
            child1, child2 = children[i]
            distance = distances[i]
            child1 = int(child1)
            child2 = int(child2)

            if child1 < len(comments):
                text1 = comments[child1]
                id1 = child1
                text1_info = None
            else:
                if child1 not in cluster_contents:
                    cluster_contents[child1] = []
                cluster_indices = cluster_contents[child1]
                cluster_size = len(cluster_indices)

                if cluster_size >= 2:
                    try:
                        local_idx1, local_idx2, _ = find_farthest_pair(
                            range(len(cluster_indices)), embeddings[cluster_indices]
                        )
                    except Exception as e:
                        print(
                            f"警告: クラスタ {child1} の処理中にエラーが発生しました: {e}"
                        )
                        local_idx1 = 0
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
                    cluster_contents[child2] = []
                cluster_indices = cluster_contents[child2]
                cluster_size = len(cluster_indices)

                if cluster_size >= 2:
                    try:
                        local_idx1, local_idx2, _ = find_farthest_pair(
                            range(len(cluster_indices)), embeddings[cluster_indices]
                        )
                    except Exception as e:
                        print(
                            f"警告: クラスタ {child2} の処理中にエラーが発生しました: {e}"
                        )
                        local_idx1 = 0
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


def save_merge_distances_csv(merges: List[Dict[str, Any]], output_dir: str) -> None:
    """併合距離のみをCSVファイルとして保存する"""
    print("併合距離をCSVファイルとして保存中...")

    import pandas as pd

    distance_data = []
    for merge in merges:
        distance_data.append(
            {"merge_index": merge["index"], "distance": merge["distance"]}
        )

    df = pd.DataFrame(distance_data)
    df.to_csv(f"{output_dir}/merge_distances.csv", index=False, encoding="utf-8")
    print(f"併合距離を {output_dir}/merge_distances.csv に保存しました")


def build_display_id(id_value, id_mapping=None):
    """IDの表示形式を「<ID>他n件」の形式に変換する"""
    if not id_mapping or id_value not in id_mapping:
        return f"ID: {id_value}"

    ids = id_mapping[id_value]
    if len(ids) == 1:
        return f"ID: {ids[0]}"
    else:
        return f"ID: {ids[0]}他{len(ids) - 1}件"


def build_display_text1(merge, id_mapping, total_comments):
    """テキスト1の表示テキストを生成する"""
    if merge["id1"] < total_comments:
        return f"テキスト1 ({build_display_id(merge['id1'], id_mapping)})"
    else:
        tid = merge["text1_info"]["text_id"]
        return f"テキスト{tid}({build_display_id(tid, id_mapping)}) from クラスタ1 (ID: {merge['id1']}, サイズ{merge['text1_info']['cluster_size']})"


def build_display_text2(merge, id_mapping, total_comments):
    """テキスト2の表示テキストを生成する"""
    if merge["id2"] < total_comments:
        return f"テキスト2 ({build_display_id(merge['id2'], id_mapping)})"
    else:
        tid = merge["text2_info"]["text_id"]
        return f"テキスト{tid}({build_display_id(tid, id_mapping)}) from クラスタ2 (ID: {merge['id2']}, サイズ{merge['text2_info']['cluster_size']})"


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

    # id_mappingの処理
    if id_mapping is None:
        id_mapping = {}

    if duplicates:
        for comment, duplicate_ids in duplicates.items():
            for id_val in duplicate_ids:
                if id_val not in id_mapping:
                    id_mapping[id_val] = []
                id_mapping[id_val].extend(duplicate_ids)

    for merge in merges:
        merge["display_h3_1"] = build_display_text1(merge, id_mapping, len(comments))
        merge["display_h3_2"] = build_display_text2(merge, id_mapping, len(comments))

    # Jinja2テンプレートを読み込む
    template_path = os.path.join(
        os.path.dirname(__file__), "templates/clustering_report.html"
    )
    template_loader = jinja2.FileSystemLoader(os.path.dirname(template_path))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(os.path.basename(template_path))

    # id_mappingの処理
    if id_mapping is None:
        id_mapping = {}

    if duplicates:
        for comment, duplicate_ids in duplicates.items():
            for id_val in duplicate_ids:
                if id_val not in id_mapping:
                    id_mapping[id_val] = []
                id_mapping[id_val].extend(duplicate_ids)

    for merge in merges:
        merge["display_h3_1"] = build_display_text1(merge, id_mapping, len(comments))
        merge["display_h3_2"] = build_display_text2(merge, id_mapping, len(comments))

    # Jinja2テンプレートを読み込む
    template_path = os.path.join(
        os.path.dirname(__file__), "templates/clustering_report.html"
    )
    template_loader = jinja2.FileSystemLoader(os.path.dirname(template_path))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(os.path.basename(template_path))
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
    sorted_duplicates = {}
    if duplicates:
        sorted_items = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_duplicates = dict(sorted_items)

        for comment, indices in sorted_duplicates.items():
            first_id = indices[0] if indices else ""
            if len(indices) > 1:
                ids_str.append(f"{first_id}他{len(indices) - 1}件")
            else:
                ids_str.append(str(first_id))
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
        duplicates=sorted_duplicates if duplicates else None,
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

    unique_comments, unique_ids, duplicates, id_mapping = remove_duplicates(
        all_comments, all_ids
    )

    comments, ids, id_mapping = get_limited_data(
        unique_comments, unique_ids, id_mapping, args.limit
    )

    comments, ids, id_mapping = get_limited_data(
        unique_comments, unique_ids, id_mapping, args.limit
    )
    # 埋め込みベクトルを生成
    embeddings = create_embeddings(comments, args.model)

    # クラスタリングを実行
    labels, clusters, distances, children = perform_clustering(
        embeddings, args.threshold
    )

    merges = extract_merge_info(
        children,
        distances,
        comments,
        embeddings,
        max_merges=-1,
        id_mapping=id_mapping,
    )
    save_merge_info(merges, comments, args.output)
    save_merge_distances_csv(merges, args.output)

    # HTMLレポートを生成
    generate_html_report(
        clusters, comments, ids, embeddings, merges, args.output, duplicates, id_mapping
    )
    print("Processing completed successfully!")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
