#!/usr/bin/env python3
"""
Simple test to verify the register_faces.py script structure without importing problematic modules.
"""

import ast
import os
import tempfile
from pathlib import Path


def test_script_syntax():
    """Test that the script has valid Python syntax."""
    print("Testing script syntax...")
    
    with open('register_faces.py', 'r') as f:
        content = f.read()
    
    try:
        ast.parse(content)
        print("  ✓ Script has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality by extracting and testing the get_image_files function separately."""
    print("\nTesting basic functionality...")
    
    # Extract the get_image_files function from the script
    with open('register_faces.py', 'r') as f:
        content = f.read()
    
    # Find the get_image_files function
    lines = content.split('\n')
    func_start = -1
    func_end = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def get_image_files(directory):'):
            func_start = i
        elif func_start != -1 and line.strip().startswith('def ') and i > func_start:
            func_end = i
            break
    
    if func_start != -1 and func_end == -1:  # Function goes to end of file
        func_end = len(lines)
    
    if func_start == -1:
        print("  ✗ Could not find get_image_files function")
        return False
    
    # Extract the function code
    func_lines = lines[func_start:func_end]
    
    # Create a test module with just the function
    test_code = '''
import os
from pathlib import Path

''' + '\n'.join(func_lines) + '''

# Test the function
if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test image files
        test_files = ['person1.jpg', 'person2.png', 'person3.jpeg', 'document.txt', 'photo.bmp']
        for filename in test_files:
            (temp_path / filename).touch()
        
        image_files = get_image_files(temp_dir)
        image_names = [f.name for f in image_files]
        
        expected_images = {'person1.jpg', 'person2.png', 'person3.jpeg', 'photo.bmp'}
        actual_images = set(image_names)
        
        print("Expected:", expected_images)
        print("Actual:", actual_images)
        
        if expected_images == actual_images:
            print("SUCCESS")
        else:
            print("FAILURE")
'''
    
    # Write test code to a temporary file and run it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(test_code)
        temp_file_path = temp_file.name
    
    try:
        import subprocess
        result = subprocess.run(['python', temp_file_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        if "SUCCESS" in result.stdout:
            print("  ✓ get_image_files function works correctly")
            return True
        else:
            print(f"  ✗ get_image_files function failed: {result.stdout} {result.stderr}")
            return False
    finally:
        os.unlink(temp_file_path)


def test_script_elements():
    """Test that the script contains required elements."""
    print("\nTesting script elements...")
    
    with open('register_faces.py', 'r') as f:
        content = f.read()
    
    # Check for required elements
    has_shebang = content.startswith('#!/usr/bin/env python3')
    has_docstring = '"""' in content[:500]  # Check beginning of file
    has_argparse_import = 'import argparse' in content
    has_async_def = 'async def main():' in content
    has_argparser = 'ArgumentParser' in content
    has_directory_arg = "'directory'" in content or '"directory"' in content
    
    elements_check = [
        ("Shebang", has_shebang),
        ("Docstring", has_docstring),
        ("Argparse import", has_argparse_import),
        ("Async main function", has_async_def),
        ("ArgumentParser", has_argparser),
        ("Directory argument", has_directory_arg)
    ]
    
    all_passed = True
    for name, passed in elements_check:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {passed}")
        if not passed:
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    print("Testing register_faces.py script structure...\n")
    
    test1 = test_script_syntax()
    test2 = test_basic_functionality()
    test3 = test_script_elements()
    
    if all([test1, test2, test3]):
        print("\n✓ All structural tests passed! The register_faces.py script is properly structured.")
        print("Note: Full functionality testing requires the model files to be available.")
    else:
        print("\n✗ Some structural tests failed.")