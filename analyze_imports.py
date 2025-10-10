#!/usr/bin/env python3
"""
Comprehensive import dependency analyzer for MyBiome back_end directory.
Extracts all imports, identifies executable scripts, and maps dependencies.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

def extract_imports_from_file(filepath):
    """Extract all imports from a Python file."""
    imports = {
        'from_imports': [],  # from X import Y
        'direct_imports': [],  # import X
        'has_main': False,  # if __name__ == "__main__"
        'errors': []
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for if __name__ == "__main__"
        if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', content):
            imports['has_main'] = True

        # Extract from X import Y statements
        from_pattern = r'from\s+([\w.]+)\s+import\s+(.+?)(?:\n|$)'
        for match in re.finditer(from_pattern, content):
            module = match.group(1)
            imported_items = match.group(2)
            imports['from_imports'].append({
                'module': module,
                'items': imported_items.strip()
            })

        # Extract import X statements
        import_pattern = r'^import\s+([\w., ]+)(?:\n|$)'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            modules = match.group(1)
            imports['direct_imports'].append(modules.strip())

    except Exception as e:
        imports['errors'].append(str(e))

    return imports

def get_module_path_from_import(import_stmt, base_path):
    """Convert import statement to potential file path."""
    # Handle back_end.src.X.Y imports
    module_parts = import_stmt.replace('back_end.', '').split('.')
    potential_paths = []

    # Try as direct file
    potential_paths.append(os.path.join(base_path, *module_parts) + '.py')

    # Try as package __init__.py
    potential_paths.append(os.path.join(base_path, *module_parts, '__init__.py'))

    return potential_paths

def analyze_directory(base_path):
    """Analyze all Python files in directory."""
    results = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, base_path)

                imports = extract_imports_from_file(filepath)
                results[rel_path] = imports

    return results

def build_dependency_graph(analysis_results, base_path):
    """Build dependency graph showing which files import which."""
    graph = {
        'imports': defaultdict(list),  # file -> list of files it imports
        'imported_by': defaultdict(list),  # file -> list of files that import it
        'executables': [],  # files with if __name__ == "__main__"
        'never_imported': [],  # files never imported anywhere
        'leaf_nodes': []  # files that import nothing
    }

    # Build imports map
    for filepath, data in analysis_results.items():
        if data['has_main']:
            graph['executables'].append(filepath)

        # Check if this is a leaf node (imports nothing internal)
        has_internal_imports = False

        for from_import in data['from_imports']:
            module = from_import['module']

            # Only track internal imports (back_end.*)
            if module.startswith('back_end.'):
                has_internal_imports = True
                # Try to find the actual file
                module_name = module.replace('back_end.src.', '').replace('back_end.', '')
                potential_file = module_name.replace('.', '/') + '.py'

                # Check if this file exists in our results
                for analyzed_file in analysis_results.keys():
                    if potential_file in analyzed_file or analyzed_file.endswith(potential_file):
                        graph['imports'][filepath].append(analyzed_file)
                        graph['imported_by'][analyzed_file].append(filepath)
                        break

        if not has_internal_imports:
            graph['leaf_nodes'].append(filepath)

    # Find files never imported
    all_files = set(analysis_results.keys())
    imported_files = set(graph['imported_by'].keys())
    graph['never_imported'] = list(all_files - imported_files)

    return graph

def identify_unused_files(analysis_results, dependency_graph):
    """Identify files that appear to be unused."""
    unused_candidates = []

    never_imported = set(dependency_graph['never_imported'])
    executables = set(dependency_graph['executables'])

    for filepath in never_imported:
        if filepath not in executables:
            # Check if it's an __init__.py (usually needed)
            if '__init__.py' in filepath:
                continue

            # This file is never imported and not executable
            unused_candidates.append({
                'file': filepath,
                'reason': 'Never imported and not executable',
                'has_main': analysis_results[filepath]['has_main'],
                'import_count': len(analysis_results[filepath]['from_imports']) +
                               len(analysis_results[filepath]['direct_imports'])
            })

    return unused_candidates

def main():
    base_path = r'c:\Users\samis\Desktop\MyBiome\back_end'

    print("Analyzing Python files in back_end directory...")
    analysis_results = analyze_directory(base_path)

    print(f"Found {len(analysis_results)} Python files")
    print("\nBuilding dependency graph...")
    dependency_graph = build_dependency_graph(analysis_results, base_path)

    print(f"Executables (has if __name__ == '__main__'): {len(dependency_graph['executables'])}")
    print(f"Leaf nodes (import nothing internal): {len(dependency_graph['leaf_nodes'])}")
    print(f"Never imported: {len(dependency_graph['never_imported'])}")

    print("\nIdentifying unused files...")
    unused_files = identify_unused_files(analysis_results, dependency_graph)

    # Prepare comprehensive report
    report = {
        'summary': {
            'total_files': len(analysis_results),
            'executables': len(dependency_graph['executables']),
            'leaf_nodes': len(dependency_graph['leaf_nodes']),
            'never_imported': len(dependency_graph['never_imported']),
            'unused_candidates': len(unused_files)
        },
        'file_details': {},
        'dependency_graph': {
            'imports': dict(dependency_graph['imports']),
            'imported_by': dict(dependency_graph['imported_by']),
            'executables': dependency_graph['executables'],
            'leaf_nodes': dependency_graph['leaf_nodes'],
            'never_imported': dependency_graph['never_imported']
        },
        'unused_candidates': unused_files
    }

    # Add detailed info for each file
    for filepath, data in analysis_results.items():
        report['file_details'][filepath] = {
            'has_main': data['has_main'],
            'from_imports_count': len(data['from_imports']),
            'direct_imports_count': len(data['direct_imports']),
            'from_imports': data['from_imports'],
            'direct_imports': data['direct_imports'],
            'is_executable': filepath in dependency_graph['executables'],
            'is_imported': filepath not in dependency_graph['never_imported'],
            'imported_by_count': len(dependency_graph['imported_by'].get(filepath, [])),
            'imports_count': len(dependency_graph['imports'].get(filepath, []))
        }

    # Save report
    output_file = r'c:\Users\samis\Desktop\MyBiome\import_analysis_report.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("UNUSED FILE CANDIDATES")
    print("="*80)
    for item in unused_files:
        print(f"\n{item['file']}")
        print(f"  Reason: {item['reason']}")
        print(f"  Has main: {item['has_main']}")
        print(f"  Import count: {item['import_count']}")

if __name__ == "__main__":
    main()
