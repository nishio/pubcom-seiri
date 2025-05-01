import re
import os

html_path = 'aipubcom_output_new/clustering_report.html'

file_size = os.path.getsize(html_path)
print(f"HTMLファイルのサイズ: {file_size / (1024*1024):.2f} MB")

def search_patterns_in_file(file_path, patterns, chunk_size=100000):
    results = {pattern: [] for pattern in patterns}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk_num = 0
        while True:
            chunk_num += 1
            print(f"チャンク {chunk_num} を処理中...")
            
            pos = f.tell()
            
            chunk = f.read(chunk_size)
            if not chunk:
                break
                
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, chunk)
                if matches:
                    results[pattern_name].extend(matches)
                    print(f"  {pattern_name}: {len(matches)}件見つかりました")
    
    return results

patterns = {
    'クラスタ情報': r'クラスタ #\d+ \(\d+件\)',
    '併合情報': r'併合 #\d+ （距離:',
    'テキスト情報': r'テキスト\d+ \(ID: \d+\)',
    'クラスタ表示形式': r'テキスト\d+\(ID: \d+\) from クラスタ\d+ \(ID: \d+, サイズ\d+\)',
    '併合タイトル': r'<div class="merge-title">併合 #\d+ （距離:',
    'クラスタ内容': r'from クラスタ\d+ \(ID: \d+, サイズ\d+\)'
}

results = search_patterns_in_file(html_path, patterns)

for pattern_name, matches in results.items():
    print(f"\n{pattern_name}の検索結果:")
    if matches:
        print(f"  合計: {len(matches)}件")
        print("  最初の5つの例:")
        for i in range(min(5, len(matches))):
            print(f"    {matches[i]}")
    else:
        print("  見つかりませんでした")

print("\nHTMLファイルの内容サンプル:")
with open(html_path, 'r', encoding='utf-8') as f:
    head_content = f.read(1000)
    print("ファイル先頭の1000文字:")
    print(head_content)
    
    f.seek(file_size // 2)
    f.readline()
    mid_content = f.read(1000)
    print("\nファイル中間の1000文字:")
    print(mid_content)
