"""Check the unified blog: links, anchors, headings, reachability, and example syntax.

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
DOCS = ROOT/'docs/ai_infra'


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
    files = sorted(DOCS.rglob('*.md'))
    # Include repository entry points and official docs that reference the series.
    for file in [*(ROOT/'docs').rglob('*.md'), *ROOT.glob('README*.md')]:
        if file not in files and (file == ROOT/'docs/README.md' or
                                  'ai_infra/' in file.read_text(encoding='utf-8-sig')):
            files.append(file)
    graph = {file.resolve(): set() for file in DOCS.rglob('*.md')}
    cache = {}
    for file in files:
        text = file.read_text(encoding='utf-8-sig')
        explicit_ids = re.findall(r'<a\s+id=["\']([^"\']+)["\']', prose(text)[0])
        if len(explicit_ids) != len(set(explicit_ids)):
            errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'duplicate-anchor'})
        _, fence = prose(text)
        if fence:
            errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'unclosed-fence'})
        if file.is_relative_to(DOCS):
            if len(re.findall(r'^#\s+.+$', prose(text)[0], re.M)) != 1:
                errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'article-title-count'})
        for raw in local_links(text):
            if not raw or re.match(r'^[A-Za-z][\w+.-]*:|^//', raw):
                continue
            parsed = urlsplit(raw)
            target = (file.parent/unquote(parsed.path)).resolve() if parsed.path else file
            link_count += 1
            if file.resolve() in graph:
                destination = target/'README.md' if target.is_dir() else target
                if destination in graph:
                    graph[file.resolve()].add(destination)
            if not target.exists():
                errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'missing-file', 'target': raw})
            elif parsed.fragment and target.suffix == '.md':
                if target not in cache:
                    cache[target] = anchors(target.read_text(encoding='utf-8-sig'))
                fragment = unquote(parsed.fragment)
                if fragment not in cache[target]:
                    errors.append({'file': file.relative_to(ROOT).as_posix(), 'kind': 'missing-anchor', 'target': raw})
    reached, pending = set(), [(DOCS/'README.md').resolve()]
    while pending:
        file = pending.pop()
        if file in reached:
            continue
        reached.add(file)
        pending.extend(graph.get(file, set()) - reached)
    for file in sorted(set(graph) - reached):
        errors.append({'kind': 'unreachable-article', 'file': file.relative_to(ROOT).as_posix()})
    for file in DOCS.rglob('*.py'):
        try:
            ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
        except SyntaxError as error:
            errors.append({'kind': 'example-syntax', 'file': str(file), 'message': str(error)})
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
