#!/usr/bin/env python3
"""
Enhanced import dependency analyzer - detects actual usage patterns.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

def normalize_path(path):
    """Normalize path for comparison."""
    return path.replace('\\', '/').replace('.py', '')

def extract_imports_and_usage(filepath):
    """Extract all imports and check for actual usage patterns."""
    data = {
        'from_imports': [],
        'direct_imports': [],
        'has_main': False,
        'imported_modules': set(),
        'errors': []
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for if __name__ == "__main__"
        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content):
            data['has_main'] = True

        # Extract from X import Y statements
        from_pattern = r'from\s+([\w.]+)\s+import\s+(.+?)(?:\n|$)'
        for match in re.finditer(from_pattern, content):
            module = match.group(1)
            imported_items = match.group(2).strip()
            data['from_imports'].append({
                'module': module,
                'items': imported_items
            })
            # Track all imported modules
            if 'back_end' in module or module.startswith('src.'):
                data['imported_modules'].add(module)

        # Extract import X statements
        import_pattern = r'^import\s+([\w., ]+)(?:\n|$)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            modules = match.group(1).strip()
            data['direct_imports'].append(modules)

    except Exception as e:
        data['errors'].append(str(e))

    data['imported_modules'] = list(data['imported_modules'])
    return data

def module_to_filepath(module_name, base_path):
    """Convert module name to potential file paths."""
    # Handle various import patterns
    module_variations = [
        module_name,
        module_name.replace('back_end.src.', ''),
        module_name.replace('back_end.', ''),
        module_name.replace('src.', ''),
    ]

    potential_files = []
    for variant in module_variations:
        parts = variant.split('.')
        # As a .py file
        potential_files.append('/'.join(parts) + '.py')
        # As a package __init__.py
        potential_files.append('/'.join(parts) + '/__init__.py')

    return potential_files

def build_import_graph(all_files_data, base_path):
    """Build actual import relationships."""
    # Normalize all file paths
    file_map = {}
    for filepath in all_files_data.keys():
        norm_path = normalize_path(filepath)
        file_map[norm_path] = filepath

    imported_by = defaultdict(list)
    imports = defaultdict(list)

    for filepath, data in all_files_data.items():
        norm_filepath = normalize_path(filepath)

        for from_imp in data['from_imports']:
            module = from_imp['module']

            # Skip non-internal imports
            if not ('back_end' in module or module.startswith('src.')):
                continue

            # Try to match to actual files
            potential_files = module_to_filepath(module, base_path)

            for pot_file in potential_files:
                # Check if this potential file matches any actual file
                for norm_file, actual_file in file_map.items():
                    if norm_file.endswith(pot_file.replace('.py', '')):
                        imports[filepath].append(actual_file)
                        imported_by[actual_file].append(filepath)
                        break

    return dict(imports), dict(imported_by)

def analyze_all_files(base_path):
    """Main analysis function."""
    all_files_data = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_path)
                all_files_data[rel_path] = extract_imports_and_usage(filepath)

    return all_files_data

def categorize_files(all_files_data, imports_graph, imported_by_graph):
    """Categorize files by usage."""
    categories = {
        'entry_points': [],  # Has __main__ and is used in orchestration
        'executables': [],  # Has __main__
        'libraries': [],  # Imported by others
        'unused': [],  # Never imported, no __main__
        'leaf_libraries': []  # Imported but imports nothing internal
    }

    all_files = set(all_files_data.keys())
    imported_files = set(imported_by_graph.keys())

    for filepath, data in all_files_data.items():
        is_imported = filepath in imported_files
        has_main = data['has_main']
        imports_others = filepath in imports_graph

        if has_main:
            # Check if it's in orchestration
            if 'orchestration' in filepath:
                categories['entry_points'].append(filepath)
            else:
                categories['executables'].append(filepath)

        if is_imported:
            if not imports_others:
                categories['leaf_libraries'].append(filepath)
            else:
                categories['libraries'].append(filepath)

        if not is_imported and not has_main and '__init__.py' not in filepath:
            categories['unused'].append(filepath)

    return categories

def main():
    base_path = r'c:\Users\samis\Desktop\MyBiome\back_end'

    print("Analyzing Python files...")
    all_files_data = analyze_all_files(base_path)
    print(f"Found {len(all_files_data)} Python files")

    print("Building import graph...")
    imports_graph, imported_by_graph = build_import_graph(all_files_data, base_path)

    print("Categorizing files...")
    categories = categorize_files(all_files_data, imports_graph, imported_by_graph)

    # Enhanced report
    report = {
        'summary': {
            'total_files': len(all_files_data),
            'entry_points': len(categories['entry_points']),
            'executables': len(categories['executables']),
            'libraries': len(categories['libraries']),
            'leaf_libraries': len(categories['leaf_libraries']),
            'unused': len(categories['unused']),
        },
        'categories': categories,
        'imports_graph': imports_graph,
        'imported_by_graph': imported_by_graph,
        'file_details': {}
    }

    # Add detailed info
    for filepath, data in all_files_data.items():
        report['file_details'][filepath] = {
            'has_main': data['has_main'],
            'from_imports_count': len(data['from_imports']),
            'direct_imports_count': len(data['direct_imports']),
            'from_imports': data['from_imports'],
            'is_imported': filepath in imported_by_graph,
            'imported_by_count': len(imported_by_graph.get(filepath, [])),
            'imports_count': len(imports_graph.get(filepath, [])),
            'imported_by': imported_by_graph.get(filepath, []),
            'imports': imports_graph.get(filepath, [])
        }

    # Save report
    output_file = r'c:\Users\samis\Desktop\MyBiome\import_analysis_v2.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total files: {report['summary']['total_files']}")
    print(f"Entry points (orchestration with __main__): {report['summary']['entry_points']}")
    print(f"Other executables: {report['summary']['executables']}")
    print(f"Library modules (imported): {report['summary']['libraries']}")
    print(f"Leaf libraries (imported but import nothing): {report['summary']['leaf_libraries']}")
    print(f"Potentially unused: {report['summary']['unused']}")

    print("\n" + "="*80)
    print("ENTRY POINTS (Main Orchestration Scripts)")
    print("="*80)
    for f in sorted(categories['entry_points']):
        imported_by = len(imported_by_graph.get(f, []))
        print(f"  {f} (imported by {imported_by} files)")

    print("\n" + "="*80)
    print("POTENTIALLY UNUSED FILES")
    print("="*80)
    for f in sorted(categories['unused']):
        imports_count = len(all_files_data[f]['from_imports']) + len(all_files_data[f]['direct_imports'])
        print(f"  {f}")
        print(f"    - Imports: {imports_count}")
        print(f"    - Never imported, no __main__")

    print("\n" + "="*80)
    print("MOST IMPORTED FILES (Top 20)")
    print("="*80)
    import_counts = [(f, len(users)) for f, users in imported_by_graph.items()]
    import_counts.sort(key=lambda x: x[1], reverse=True)
    for f, count in import_counts[:20]:
        print(f"  {f}: {count} files import this")

if __name__ == "__main__":
    main()
