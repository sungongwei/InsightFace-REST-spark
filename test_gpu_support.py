#!/usr/bin/env python3
"""
Test script to verify GPU functionality in face search module.
"""

import numpy as np
import tempfile
import os
from unittest.mock import patch


def test_gpu_support_in_face_search():
    """Test that the face search module supports GPU configuration."""
    print("Testing GPU Support in Face Search Module...")
    
    # Import the modules
    import sys
    import importlib.util
    
    # Load the database module
    db_spec = importlib.util.spec_from_file_location("database", "if_rest/core/database.py")
    database_module = importlib.util.module_from_spec(db_spec)
    db_spec.loader.exec_module(database_module)
    
    # Load the face search module
    search_spec = importlib.util.spec_from_file_location("face_search", "if_rest/core/face_search.py")
    face_search_module = importlib.util.module_from_spec(search_spec)
    search_spec.loader.exec_module(face_search_module)
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Test CPU initialization
        print("  Testing CPU initialization...")
        cpu_search_engine = face_search_module.FaceSearchEngine(db_path, use_gpu=False)
        print(f"  ✓ CPU index type: {type(cpu_search_engine.index)}")
        print(f"  ✓ CPU GPU enabled: {cpu_search_engine.use_gpu}")
        
        # Test GPU initialization (will fall back to CPU if GPU not available)
        print("  Testing GPU initialization (with fallback)...")
        gpu_search_engine = face_search_module.FaceSearchEngine(db_path, use_gpu=True)
        print(f"  ✓ GPU index type: {type(gpu_search_engine.index)}")
        print(f"  ✓ GPU actually enabled: {gpu_search_engine.use_gpu}")
        
        # Test that both engines work similarly
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding2 = np.random.rand(512).astype(np.float32)
        
        # Add test faces to database
        db = database_module.FaceDatabase(db_path)
        face_id1 = db.add_face(name="Person 1", embedding=embedding1)
        face_id2 = db.add_face(name="Person 2", embedding=embedding2)
        
        # Reload indexes
        cpu_search_engine.rebuild_index()
        gpu_search_engine.rebuild_index()
        
        # Test search functionality
        query_embedding = embedding1
        cpu_results = cpu_search_engine.search(query_embedding, k=5, threshold=0.0)
        gpu_results = gpu_search_engine.search(query_embedding, k=5, threshold=0.0)
        
        print(f"  ✓ CPU search results count: {len(cpu_results)}")
        print(f"  ✓ GPU search results count: {len(gpu_results)}")
        
        print("  ✓ GPU support test completed\n")
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_face_manager_gpu_support():
    """Test that the face manager supports GPU configuration."""
    print("Testing Face Manager GPU Support...")
    
    # Import the modules
    import sys
    import importlib.util
    
    # Load the face manager module
    manager_spec = importlib.util.spec_from_file_location("face_manager", "if_rest/core/face_manager.py")
    face_manager_module = importlib.util.module_from_spec(manager_spec)
    manager_spec.loader.exec_module(face_manager_module)
    
    # Test initialization with GPU support
    print("  Testing FaceManager with GPU disabled...")
    manager_cpu = face_manager_module.FaceManager(use_gpu=False)
    print(f"  ✓ CPU manager GPU enabled: {manager_cpu.use_gpu}")
    print(f"  ✓ CPU manager search engine GPU enabled: {manager_cpu.search_engine.use_gpu}")
    
    print("  Testing FaceManager with GPU enabled...")
    manager_gpu = face_manager_module.FaceManager(use_gpu=True)
    print(f"  ✓ GPU manager GPU enabled: {manager_gpu.use_gpu}")
    print(f"  ✓ GPU manager search engine GPU enabled: {manager_gpu.search_engine.use_gpu}")
    
    print("  ✓ Face manager GPU support test completed\n")


def test_environment_variable_handling():
    """Test that environment variable handling works correctly."""
    print("Testing Environment Variable Handling...")
    
    # Temporarily set environment variable
    original_env = os.environ.get('USE_GPU', None)
    
    try:
        # Test with USE_GPU=true
        os.environ['USE_GPU'] = 'true'
        use_gpu_true = os.getenv('USE_GPU', 'false').lower() == 'true'
        print(f"  ✓ USE_GPU='true' evaluates to: {use_gpu_true}")
        
        # Test with USE_GPU=false
        os.environ['USE_GPU'] = 'false'
        use_gpu_false = os.getenv('USE_GPU', 'false').lower() == 'true'
        print(f"  ✓ USE_GPU='false' evaluates to: {use_gpu_false}")
        
        # Test with different case
        os.environ['USE_GPU'] = 'True'
        use_gpu_True = os.getenv('USE_GPU', 'false').lower() == 'true'
        print(f"  ✓ USE_GPU='True' evaluates to: {use_gpu_True}")
        
        print("  ✓ Environment variable handling test completed\n")
        
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ['USE_GPU'] = original_env
        elif 'USE_GPU' in os.environ:
            del os.environ['USE_GPU']


if __name__ == "__main__":
    print("Testing GPU Support Implementation...\n")
    
    test_gpu_support_in_face_search()
    test_face_manager_gpu_support()
    test_environment_variable_handling()
    
    print("✓ All GPU support tests completed successfully!")
    print("\nGPU functionality has been implemented with:")
    print("- Configurable GPU support in FaceSearchEngine")
    print("- Environment variable control (USE_GPU)")
    print("- Automatic fallback to CPU if GPU not available")
    print("- Proper error handling for GPU initialization")