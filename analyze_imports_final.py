#!/usr/bin/env python3
"""
Final comprehensive import analysis with relative import support.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

def extract_all_imports(filepath):
    """Extract all import types including relative imports."""
    data = {
        'from_imports': [],
        'direct_imports': [],
        'relative_imports': [],
        'has_main': False,
        'errors': []
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content):
            data['has_main'] = True

        # Relative imports: from . import X or from .module import Y
        relative_pattern = r'from\s+(\.\S*)\s+import\s+(.+?)(?:\n|$)'
        for match in re.finditer(relative_pattern, content):
            module = match.group(1)
            items = match.group(2).strip()
            data['relative_imports'].append({
                'module': module,
                'items': items
            })

        # Regular from imports
        from_pattern = r'from\s+([\w.]+)\s+import\s+(.+?)(?:\n|$)'
        for match in re.finditer(from_pattern, content):
            module = match.group(1)
            if not module.startswith('.'):  # Skip relative (handled above)
                items = match.group(2).strip()
                data['from_imports'].append({
                    'module': module,
                    'items': items
                })

        # Direct imports
        import_pattern = r'^import\s+([\w., ]+)(?:\n|$)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            modules = match.group(1).strip()
            data['direct_imports'].append(modules)

    except Exception as e:
        data['errors'].append(str(e))

    return data

def resolve_relative_import(filepath, relative_module):
    """Resolve relative import to absolute path."""
    # Get directory of current file
    current_dir = os.path.dirname(filepath)

    # Count leading dots
    dots = len(relative_module) - len(relative_module.lstrip('.'))
    module_name = relative_module.lstrip('.')

    # Navigate up directories based on dot count
    target_dir = current_dir
    for _ in range(dots - 1):
        target_dir = os.path.dirname(target_dir)

    # Build potential file paths
    if module_name:
        # from .module import X
        potential_file = os.path.join(target_dir, module_name.replace('.', os.sep) + '.py')
        potential_init = os.path.join(target_dir, module_name.replace('.', os.sep), '__init__.py')
    else:
        # from . import X (imports from __init__.py)
        potential_file = os.path.join(target_dir, '__init__.py')
        potential_init = None

    return [potential_file, potential_init] if potential_init else [potential_file]

def normalize_path(path, base_path):
    """Normalize path relative to base_path."""
    try:
        rel_path = os.path.relpath(path, base_path)
        return rel_path.replace('\\', '/')
    except:
        return path

def build_complete_graph(all_files_data, base_path):
    """Build complete import graph including relative imports."""
    # Create normalized path mapping
    norm_to_actual = {}
    for filepath in all_files_data.keys():
        abs_path = os.path.join(base_path, filepath)
        norm_path = normalize_path(abs_path, base_path)
        norm_to_actual[norm_path] = filepath

    imports_graph = defaultdict(list)
    imported_by_graph = defaultdict(list)

    for filepath, data in all_files_data.items():
        abs_filepath = os.path.join(base_path, filepath)

        # Handle relative imports
        for rel_import in data['relative_imports']:
            module = rel_import['module']
            potential_paths = resolve_relative_import(abs_filepath, module)

            for pot_path in potential_paths:
                norm_pot = normalize_path(pot_path, base_path)
                if norm_pot in norm_to_actual:
                    target_file = norm_to_actual[norm_pot]
                    if target_file not in imports_graph[filepath]:
                        imports_graph[filepath].append(target_file)
                    if filepath not in imported_by_graph[target_file]:
                        imported_by_graph[target_file].append(filepath)

        # Handle absolute imports (back_end.src.*)
        for from_imp in data['from_imports']:
            module = from_imp['module']

            if not ('back_end' in module or module.startswith('src.')):
                continue

            # Convert module path to file path
            module_clean = module.replace('back_end.src.', '').replace('back_end.', '').replace('src.', '')
            parts = module_clean.split('.')

            # Try as .py file
            potential_file = os.path.join(base_path, 'src', *parts) + '.py'
            potential_init = os.path.join(base_path, 'src', *parts, '__init__.py')

            for pot_path in [potential_file, potential_init]:
                norm_pot = normalize_path(pot_path, base_path)
                if norm_pot in norm_to_actual:
                    target_file = norm_to_actual[norm_pot]
                    if target_file not in imports_graph[filepath]:
                        imports_graph[filepath].append(target_file)
                    if filepath not in imported_by_graph[target_file]:
                        imported_by_graph[target_file].append(filepath)

    return dict(imports_graph), dict(imported_by_graph)

def analyze_all_files(base_path):
    """Analyze all Python files."""
    all_files_data = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_path).replace('\\', '/')
                all_files_data[rel_path] = extract_all_imports(filepath)

    return all_files_data

def categorize_files(all_files_data, imports_graph, imported_by_graph):
    """Categorize files by usage patterns."""
    categories = {
        'entry_points': [],
        'executables': [],
        'libraries': [],
        'unused': [],
        'init_files': []
    }

    all_files = set(all_files_data.keys())
    imported_files = set(imported_by_graph.keys())

    for filepath, data in all_files_data.items():
        is_imported = filepath in imported_files
        has_main = data['has_main']
        is_init = '__init__.py' in filepath

        if is_init:
            categories['init_files'].append(filepath)
        elif has_main and 'orchestration' in filepath:
            categories['entry_points'].append(filepath)
        elif has_main:
            categories['executables'].append(filepath)
        elif is_imported:
            categories['libraries'].append(filepath)
        else:
            # Truly unused: not imported, no __main__, not __init__
            categories['unused'].append(filepath)

    return categories

def identify_duplicates(all_files_data):
    """Identify potential duplicate functionality."""
    duplicates = []

    # Check for similar file names in different locations
    name_to_paths = defaultdict(list)
    for filepath in all_files_data.keys():
        basename = os.path.basename(filepath)
        if basename != '__init__.py':
            name_to_paths[basename].append(filepath)

    for basename, paths in name_to_paths.items():
        if len(paths) > 1:
            duplicates.append({
                'name': basename,
                'paths': paths,
                'reason': 'Same filename in multiple locations'
            })

    return duplicates

def main():
    base_path = r'c:\Users\samis\Desktop\MyBiome\back_end'

    print("Analyzing all Python files...")
    all_files_data = analyze_all_files(base_path)
    print(f"Found {len(all_files_data)} Python files")

    print("Building complete import graph (including relative imports)...")
    imports_graph, imported_by_graph = build_complete_graph(all_files_data, base_path)

    print("Categorizing files...")
    categories = categorize_files(all_files_data, imports_graph, imported_by_graph)

    print("Identifying duplicates...")
    duplicates = identify_duplicates(all_files_data)

    # Build report
    report = {
        'summary': {
            'total_files': len(all_files_data),
            'entry_points': len(categories['entry_points']),
            'executables': len(categories['executables']),
            'libraries': len(categories['libraries']),
            'unused': len(categories['unused']),
            'init_files': len(categories['init_files']),
            'duplicates': len(duplicates)
        },
        'categories': categories,
        'duplicates': duplicates,
        'file_details': {}
    }

    # Add detailed info
    for filepath, data in all_files_data.items():
        report['file_details'][filepath] = {
            'has_main': data['has_main'],
            'from_imports_count': len(data['from_imports']),
            'relative_imports_count': len(data['relative_imports']),
            'direct_imports_count': len(data['direct_imports']),
            'is_imported': filepath in imported_by_graph,
            'imported_by_count': len(imported_by_graph.get(filepath, [])),
            'imports_count': len(imports_graph.get(filepath, [])),
            'imported_by': imported_by_graph.get(filepath, []),
            'imports': imports_graph.get(filepath, []),
            'relative_imports': data['relative_imports']
        }

    # Save report
    output_file = r'c:\Users\samis\Desktop\MyBiome\import_analysis_final.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("FINAL ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total files: {report['summary']['total_files']}")
    print(f"Entry points (orchestration): {report['summary']['entry_points']}")
    print(f"Other executables: {report['summary']['executables']}")
    print(f"Library modules: {report['summary']['libraries']}")
    print(f"Init files: {report['summary']['init_files']}")
    print(f"UNUSED files: {report['summary']['unused']}")
    print(f"Potential duplicates: {report['summary']['duplicates']}")

    print("\n" + "="*80)
    print("ENTRY POINTS")
    print("="*80)
    for f in sorted(categories['entry_points']):
        print(f"  {f}")

    print("\n" + "="*80)
    print("TRULY UNUSED FILES")
    print("="*80)
    for f in sorted(categories['unused']):
        print(f"  {f}")
        imports = len(all_files_data[f]['from_imports']) + len(all_files_data[f]['relative_imports'])
        print(f"    - Total imports: {imports}")

    print("\n" + "="*80)
    print("POTENTIAL DUPLICATES")
    print("="*80)
    for dup in duplicates:
        print(f"\n{dup['name']}:")
        for path in dup['paths']:
            print(f"  - {path}")

    print("\n" + "="*80)
    print("MOST IMPORTED FILES (Top 15)")
    print("="*80)
    import_counts = [(f, len(users)) for f, users in imported_by_graph.items()]
    import_counts.sort(key=lambda x: x[1], reverse=True)
    for f, count in import_counts[:15]:
        print(f"  {count:3d} imports: {f}")

if __name__ == "__main__":
    main()
