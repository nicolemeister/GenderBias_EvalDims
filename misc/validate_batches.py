#!/usr/bin/env python3
"""
Script to check output data structure and identify violations.

For each model in processed_batch_outputs, checks:
1. Each job_name folder should have 4 subfolders (or 2 if job_bundle is zollo)
2. Each subfolder should have 1000 files
"""

import os
import re
import csv
from pathlib import Path
from typing import List, Tuple, Optional, Dict


def parse_folder_name(folder_name: str) -> Optional[Tuple[str, str]]:
    """
    Parse folder name to extract name_bundle and job_bundle.
    
    Example: temp_1.0_names_armstrong_jobresumes_rozado
    -> name_bundle=armstrong, job_bundle=rozado
    
    Returns: (name_bundle, job_bundle) or None if parsing fails
    """
    # Pattern: temp_*_names_{name_bundle}_jobresumes_{job_bundle}
    pattern = r'temp_[^_]+_names_([^_]+)_jobresumes_([^_]+)'
    match = re.match(pattern, folder_name)
    
    if match:
        name_bundle = match.group(1)
        job_bundle = match.group(2)
        return (name_bundle, job_bundle)
    
    return None


def count_files_in_folder(folder_path: Path) -> int:
    """Count the number of files (not directories) in a folder."""
    try:
        return len([f for f in folder_path.iterdir() if f.is_file()])
    except (PermissionError, OSError) as e:
        print(f"Error accessing {folder_path}: {e}")
        return 0


def check_model_structure(base_path: Path, model_name: str) -> List[Dict[str, str]]:
    """
    Check structure for a single model.
    
    Returns list of violation records.
    """
    violations = []
    model_path = base_path / model_name
    
    if not model_path.exists():
        return violations
    
    # Iterate through each job_name folder
    for job_name_folder in model_path.iterdir():
        if not job_name_folder.is_dir():
            continue
        
        # Parse folder name to get name_bundle and job_bundle
        parsed = parse_folder_name(job_name_folder.name)
        if parsed is None:
            # If parsing fails, log as violation
            violations.append({
                'model': model_name,
                'name_bundle': 'N/A',
                'job_bundle': 'N/A',
                'job_name': job_name_folder.name,
                'num_files': 'PARSE_ERROR'
            })
            continue
        
        name_bundle, job_bundle = parsed
        
        # Get all subdirectories (job_name folders)
        subdirs = [d for d in job_name_folder.iterdir() if d.is_dir()]
        num_subdirs = len(subdirs)
        
        # Expected number of subdirectories
        expected_num_subdirs = 2 if job_bundle == 'zollo' else 4
        
        # Check if we have the correct number of subdirectories
        if num_subdirs != expected_num_subdirs:
            violations.append({
                'model': model_name,
                'name_bundle': name_bundle,
                'job_bundle': job_bundle,
                'job_name': job_name_folder.name,
                'num_files': f'Expected {expected_num_subdirs} folders, found {num_subdirs}'
            })
        
        # Check each subdirectory for file count
        for subdir in subdirs:
            num_files = count_files_in_folder(subdir)
            if num_files < 900:
                violations.append({
                    'model': model_name,
                    'name_bundle': name_bundle,
                    'job_bundle': job_bundle,
                    'job_name': subdir.name,
                    'num_files': num_files
                })
    
    return violations


def main():
    """Main function to check all models and generate violation report."""
    base_path = Path('/nlp/scr/nmeist/EvalDims/output_data/yin/together/processed_batch_outputs')
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    all_violations = []
    
    # Iterate through each model
    for model_folder in base_path.iterdir():
        if not model_folder.is_dir():
            continue
        
        model_name = model_folder.name
        print(f"Checking model: {model_name}")
        
        violations = check_model_structure(base_path, model_name)
        all_violations.extend(violations)
    
    # Write to CSV
    output_path = Path('/nlp/scr/nmeist/EvalDims/misc/output_structure_violations.csv')
    
    if all_violations:
        # Write violations to CSV
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['model', 'name_bundle', 'job_bundle', 'job_name', 'num_files']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_violations)
        
        print(f"\nFound {len(all_violations)} violations.")
        print(f"Results saved to: {output_path}")
        print("\nFirst few violations:")
        for i, violation in enumerate(all_violations[:5], 1):
            print(f"{i}. {violation}")
    else:
        print("\nNo violations found! All structures are correct.")
        # Still create an empty CSV with the correct columns
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['model', 'name_bundle', 'job_bundle', 'job_name', 'num_files']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Empty report saved to: {output_path}")


if __name__ == '__main__':
    main()
