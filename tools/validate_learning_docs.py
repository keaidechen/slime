"""Check the learning corpus' local Markdown links, anchors, and migration records.

Run from any directory: python tools/validate_learning_docs.py
GPU/framework examples are not executed by this validator.
"""
from pathlib import Path
import argparse
import ast
import json
import re
import unicodedata
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DIRS = ('learn_docs', 'docs/code_walkthrough', 'docs/megatron_code_walkthrough',
        'docs/paper_read', 'docs/performance_analysis_guide',
        'docs/sglang_code_walkthrough', 'docs/runtime_comparisons')


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^\s*(?:>\s*)?(`{3,}|~{3,})', line)
        if match:
            marker = match[1][0]
            if fence == marker:
                fence = None
            elif fence is None:
                fence = marker
            lines.append('')
        elif fence is None:
            lines.append(line)
        else:
            lines.append('')
    return '\n'.join(lines), fence


def slug(value):
    value = re.sub(r'<[^>]+>', '', value).lower()
    # GitHub-style heading slug: preserve letters/numbers, hyphen, underscore.
    value = ''.join(ch for ch in value if ch in '-_ ' or
                    unicodedata.category(ch)[0] in ('L', 'N', 'M'))
    return value.replace(' ', '-')


def anchors(text):
    text, _ = prose(text)
    result = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', text))
    counts = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        key = slug(heading)
        count = counts.get(key, 0)
        counts[key] = count + 1
        result.add(key + (f'-{count}' if count else ''))
    return result


def local_links(text):
    text, _ = prose(text)
    # Inline code frequently describes Markdown paths/examples rather than links.
    text = re.sub(r'`[^`\n]*`', '', text)
    for match in re.finditer(r'\]\((<[^>\n]+>|[^)\n]+)\)', text):
        target = match[1]
        if target.startswith('<') and target.endswith('>'):
            target = target[1:-1]
        else:
            target = re.split(r'\s+["\']', target, maxsplit=1)[0]
        yield target
    for match in re.finditer(r'^\s*\[[^\]]+\]:\s*(\S+)', text, re.M):
        yield match[1].strip('<>')


def audit():
    errors, link_count = [], 0
    files = sorted({p for directory in DIRS for p in (ROOT/directory).rglob('*.md')})
    cache = {}
    for file in files:
        text = file.read_text(encoding='utf-8-sig')
        explicit_ids = re.findall(r'<a\s+id=["\']([^"\']+)["\']', prose(text)[0])
        if len(explicit_ids) != len(set(explicit_ids)):
            errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'duplicate-anchor'})
        _, fence = prose(text)
        if fence:
            errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'unclosed-fence'})
        for raw in local_links(text):
            if not raw or re.match(r'^[A-Za-z][\w+.-]*:|^//', raw):
                continue
            parsed = urlsplit(raw)
            target = (file.parent/unquote(parsed.path)).resolve() if parsed.path else file
            link_count += 1
            if not target.exists():
                errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'missing-file', 'target': raw})
            elif parsed.fragment and target.suffix == '.md':
                if target not in cache:
                    cache[target] = anchors(target.read_text(encoding='utf-8-sig'))
                fragment = unquote(parsed.fragment)
                if fragment not in cache[target]:
                    errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'missing-anchor', 'target': raw})
    manifest_path = ROOT/'learn_docs/document_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        docs = manifest['documents']
        if len(docs) != manifest['baseline_documents'] or len({d['source'] for d in docs}) != len(docs):
            errors.append({'kind': 'manifest-coverage'})
        removed = manifest.get('removed_files', [])
        if len(removed) != len(set(removed)):
            errors.append({'kind': 'duplicate-removal'})
        for path in removed:
            if (ROOT/path).exists():
                errors.append({'kind': 'removed-file-still-exists', 'target': path})
        for doc in docs:
            expected = 'removed' if doc['source'] in removed else 'present'
            if doc.get('source_status', 'present') != expected:
                errors.append({'kind': 'manifest-source-status', 'target': doc['source']})
            if not doc['current']:
                errors.append({'kind': 'manifest-no-destination', 'target': doc['source']})
            paths = doc['current'] if expected == 'removed' else [doc['source'], *doc['current']]
            for path in paths:
                if not (ROOT/path).is_file():
                    errors.append({'kind': 'manifest-missing-file', 'target': path})
    routes_path = ROOT/'learn_docs/document_routes.json'
    if routes_path.exists():
        routes = json.loads(routes_path.read_text(encoding='utf-8'))
        if manifest_path.exists() and set(routes['routes']) != set(manifest.get('removed_files', [])):
            errors.append({'kind': 'removal-route-coverage'})
        for source, targets in routes['routes'].items():
            if not targets:
                errors.append({'kind': 'route-no-destination', 'target': source})
            for row in targets:
                target = ROOT/row['path']
                if not target.is_file():
                    errors.append({'kind': 'route-missing-file', 'target': row['path']})
                elif row.get('anchor') and row['anchor'] not in anchors(target.read_text(encoding='utf-8')):
                    errors.append({'kind': 'route-missing-anchor', 'target': row['path'], 'anchor': row['anchor']})
        for row in routes['merges']:
            target = ROOT/row['destination']
            if not target.is_file():
                errors.append({'kind': 'migration-missing-file', 'target': row['destination']})
            elif row.get('anchor') and row['anchor'] not in anchors(target.read_text(encoding='utf-8')):
                errors.append({'kind': 'migration-missing-anchor', 'target': row['destination'],
                               'anchor': row['anchor']})
    for file in (ROOT/'learn_docs/labs').glob('*.py'):
        try:
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
        except SyntaxError as error:
            errors.append({'kind': 'lab-syntax', 'file': str(file), 'message': str(error)})
    unique = list({json.dumps(e, sort_keys=True): e for e in errors}.values())
    return {'markdown_files': len(files), 'local_links': link_count,
            'errors': unique, 'error_count': len(unique)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    report = audit()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print({k: v for k, v in report.items() if k != 'errors'})
        for error in report['errors']:
            print(error)
    raise SystemExit(bool(report['error_count']))


if __name__ == '__main__':
    main()
