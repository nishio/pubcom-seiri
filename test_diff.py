import difflib

text1 = "これはテストです"
text2 = "これはテスト文です"

d = difflib.Differ()
diff = list(d.compare(text1, text2))
print("Differ output:")
print(diff)

s = difflib.SequenceMatcher(None, text1, text2)
opcodes = list(s.get_opcodes())
print("\nSequenceMatcher opcodes:")
print(opcodes)

def generate_html_diff_old(text1, text2):
    d = difflib.Differ()
    diff = list(d.compare(text1, text2))
    
    html = []
    for line in diff:
        if line.startswith('  '):
            html.append(f'<span class="common">{line[2:]}</span>')
        elif line.startswith('- '):
            html.append(f'<span class="removed">{line[2:]}</span>')
        elif line.startswith('+ '):
            html.append(f'<span class="added">{line[2:]}</span>')
    
    return ''.join(html)

def generate_html_diff_new(text1, text2):
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

def generate_char_diff_html(text1, text2):
    s = difflib.SequenceMatcher(None, text1, text2)
    result = []
    
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            result.append(f'<span class="common">{text1[i1:i2]}</span>')
        elif tag == 'delete':
            result.append(f'<span class="removed">{text1[i1:i2]}</span>')
        elif tag == 'insert':
            result.append(f'<span class="added">{text2[j1:j2]}</span>')
        elif tag == 'replace':
            result.append(f'<span class="removed">{text1[i1:i2]}</span>')
            result.append(f'<span class="added">{text2[j1:j2]}</span>')
    
    return ''.join(result)

print("\nOld HTML output:")
print(generate_html_diff_old(text1, text2))

print("\nNew HTML output (optimized):")
print(generate_html_diff_new(text1, text2))

print("\nCharacter-level diff HTML output:")
print(generate_char_diff_html(text1, text2))
