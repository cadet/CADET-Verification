import os
import json
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re


def get_github_files(url: str) -> List[str]:
    """
    Get list of files from a GitHub repository URL.
    Converts GitHub URLs to API format and recursively gets all files.
    """
    # Convert GitHub URL to API URL
    if 'github.com' in url:
        # Parse GitHub URL: https://github.com/user/repo/tree/branch/path
        parts = url.split('github.com/')[-1].split('/')
        if len(parts) >= 2:
            user, repo = parts[0], parts[1]
            branch = 'main'  # default branch
            path = ''
            
            if len(parts) > 2 and parts[2] == 'tree':
                if len(parts) > 3:
                    branch = parts[3]
                if len(parts) > 4:
                    path = '/'.join(parts[4:])
            
            api_url = f"https://api.github.com/repos/{user}/{repo}/contents/{path}"
            return get_github_files_recursive(api_url, branch)
    
    return []


def get_github_files_recursive(api_url: str, branch: str = 'main') -> List[str]:
    """Recursively get all files from GitHub API"""
    files = []
    try:
        response = requests.get(api_url, params={'ref': branch})
        response.raise_for_status()
        
        for item in response.json():
            if item['type'] == 'file':
                files.append(item['download_url'])
            elif item['type'] == 'dir':
                # Recursively get files from subdirectories
                files.extend(get_github_files_recursive(item['url'], branch))
    except Exception as e:
        print(f"Error accessing GitHub API: {e}")
    
    return files


def get_local_files(directory: str) -> List[str]:
    """Get all files from local directory recursively"""
    files = []
    try:
        for root, dirs, filenames in os.walk(directory):
            for filename in filenames:
                files.append(os.path.join(root, filename))
    except Exception as e:
        print(f"Error accessing local directory {directory}: {e}")
    
    return files


def get_files_from_path(path: str) -> List[str]:
    """Get files from either local directory or remote URL"""
    if path.startswith(('http://', 'https://')):
        return get_github_files(path)
    else:
        return get_local_files(path)


def filter_convergence_json(files: List[str]) -> List[str]:
    """Filter files to only include JSON files starting with 'convergence'"""
    filtered = []
    for file in files:
        filename = os.path.basename(file)
        if filename.lower().startswith('convergence') and filename.lower().endswith('.json'):
            filtered.append(file)
    return filtered


def get_file_content(file_path: str) -> Dict:
    """Get JSON content from file (local or remote)"""
    try:
        if file_path.startswith(('http://', 'https://')):
            response = requests.get(file_path)
            response.raise_for_status()
            return response.json()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}


def compare_values(val1, val2, key_path: str) -> bool:
    """Compare two values recursively, return True if equal"""
    if type(val1) != type(val2):
        print(f"ERROR: Type mismatch at '{key_path}': {type(val1).__name__} vs {type(val2).__name__}")
        return False
    
    if isinstance(val1, dict):
        if set(val1.keys()) != set(val2.keys()):
            missing_in_1 = set(val2.keys()) - set(val1.keys())
            missing_in_2 = set(val1.keys()) - set(val2.keys())
            if missing_in_1:
                print(f"ERROR: Keys missing in first file at '{key_path}': {missing_in_1}")
            if missing_in_2:
                print(f"ERROR: Keys missing in second file at '{key_path}': {missing_in_2}")
            return False
        
        for key in val1:
            if key == 'Sim. time':
                try:
                    # Convert to float lists if they aren't already
                    time_list_1 = [float(x) for x in val1[key]]
                    time_list_2 = [float(x) for x in val2[key]]
                    
                    if len(time_list_1) == len(time_list_2):
                        differences = calculate_percentage_difference(time_list_1, time_list_2)
                        if not all(x == 0.0 for x in differences):
                            print(f"Sim. time percentage differences: {differences}")
                    else:
                        print(f"ERROR: Sim. time lists have different lengths: {len(time_list_1)} vs {len(time_list_2)}")
                except (ValueError, TypeError) as e:
                    print(f"ERROR: Could not convert Sim. time values to numbers: {e}")
                return True
            else:
                if not compare_values(val1[key], val2[key], f"{key_path}.{key}"):
                    return False
        return True
    
    elif isinstance(val1, list):
        if len(val1) != len(val2):
            print(f"ERROR: List length mismatch at '{key_path}': {len(val1)} vs {len(val2)}")
            return False
        
        for i, (item1, item2) in enumerate(zip(val1, val2)):
            if not compare_values(item1, item2, f"{key_path}[{i}]"):
                return False
        return True
    
    else:
        if val1 != val2:
            print(f"ERROR: Value mismatch at '{key_path}': {val1} vs {val2}")
            return False
        return True


def calculate_percentage_difference(list1: List[float], list2: List[float]) -> List[float]:
    """Calculate percentage difference between two lists of numbers"""
    if len(list1) != len(list2):
        return []
    
    differences = []
    for val1, val2 in zip(list1, list2):
        if val2 != 0:
            diff_percent = ((val1 - val2) / val2) * 100
        elif val1 != 0:
            diff_percent = float('inf')  # or handle as needed
        else:
            diff_percent = 0.0
        differences.append(diff_percent)
    
    return differences


def compare_json_directories(dir1: str, dir2: str):
    """
    Main function to compare JSON files between two directories.
    
    Args:
        dir1: First directory path (local or URL)
        dir2: Second directory path (local or URL)
    """
    print(f"Comparing directories:")
    print(f"  Directory 1: {dir1}")
    print(f"  Directory 2: {dir2}")
    print("=" * 60)
    
    # Get all files from both directories
    files1 = get_files_from_path(dir1)
    files2 = get_files_from_path(dir2)
    
    # Filter for convergence JSON files
    json_files1 = filter_convergence_json(files1)
    json_files2 = filter_convergence_json(files2)
    
    print(f"Found {len(json_files1)} convergence JSON files in directory 1")
    print(f"Found {len(json_files2)} convergence JSON files in directory 2")
    
    # Extract just the filenames for comparison
    names1 = {os.path.basename(f) for f in json_files1}
    names2 = {os.path.basename(f) for f in json_files2}
    
    # Find files that exist in only one directory
    only_in_dir1 = names1 - names2
    only_in_dir2 = names2 - names1
    common_files = names1 & names2
    
    print("\n" + "=" * 60)
    print("FILES EXISTING IN ONLY ONE DIRECTORY:")
    print("=" * 60)
    
    if only_in_dir1:
        print(f"\nFiles only in directory 1 ({len(only_in_dir1)} files):")
        for filename in sorted(only_in_dir1):
            print(f"  - {filename}")
    
    if only_in_dir2:
        print(f"\nFiles only in directory 2 ({len(only_in_dir2)} files):")
        for filename in sorted(only_in_dir2):
            print(f"  - {filename}")
    
    if not only_in_dir1 and not only_in_dir2:
        print("  No files exist in only one directory.")
    
    print("\n" + "=" * 60)
    print("COMPARING COMMON FILES:")
    print("=" * 60)
    
    # Compare common files
    for filename in sorted(common_files):   
        
        print(f"\nComparing: {filename}")
        print("-" * 40)
        
        # Find full paths for the files
        file1_path = next((f for f in json_files1 if os.path.basename(f) == filename), None)
        file2_path = next((f for f in json_files2 if os.path.basename(f) == filename), None)
        
        if not file1_path or not file2_path:
            print(f"ERROR: Could not find full path for {filename}")
            continue
        
        # Load JSON content
        content1 = get_file_content(file1_path)
        content2 = get_file_content(file2_path)
        
        if not content1 or not content2:
            print(f"ERROR: Could not load content from one or both files")
            continue
        
        content1 = content1.get('convergence', None)
        content2 = content2.get('convergence', None)
        
        if content1 is None or content2 is None:
            print(f"ERROR: one of the two files has no key 'convergence'")
            continue
        
        content2Keys = list(content2.keys())
        
        for key1 in content1.keys():
            
            key2 = content2.get(key1, None)
            if key2 is None:
                print(f"ERROR: {key1} does not exist in {file2_path}")
                continue
            
            content2Keys.remove(key1)
            
            files_match = compare_values(content1, content2, key1)
        
        if len(content2Keys):
            print(f"ERROR: {content2Keys} do not exist in {file1_path}")
            continue
        
        if not files_match:
            print("✗ Files have differences")
        else:
            print("✓ Files match (excluding Sim. time)")


# # Example usage:

# compare_json_directories(
#     r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\2Dchromatography",
#     r"C:\Users\jmbr\software\CADET-Verification\output\test_cadet-core\test_2Dchromatography")


# compare_json_directories(  
# r"https://github.com/cadet/CADET-Verification-Output/tree/2024-10-25_09-25-00_main_5030418/test_cadet-core",
# r"https://github.com/cadet/CADET-Verification-Output/tree/2024-12-12_15-38-58_main_6ed1238/test_cadet-core"
# )
  
  