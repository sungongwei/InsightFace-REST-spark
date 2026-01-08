#!/usr/bin/env python3
"""
Script to register images from a directory to the face database.
The filename (without extension) will be used as the face name.
"""

import argparse
import os
import sys
from pathlib import Path
import asyncio
import cv2
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from if_rest.core.database import FaceDatabase
from if_rest.core.face_search import FaceSearchEngine
from if_rest.core.face_manager import FaceManager
from if_rest.core.processing import get_processing


def get_image_files(directory):
    """Get all image files from the specified directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


async def process_image_file(image_path, face_manager, verbose=False):
    """Process a single image file and add it to the database."""
    try:
        # Extract face name from filename (without extension)
        face_name = image_path.stem
        if verbose:
            print(f"Processing: {image_path.name} -> Face name: '{face_name}'")
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Add face to database
        result = await face_manager.add_face_from_image(
            name=face_name,
            image_data=image_data,
            extract_embedding=True
        )
        
        if result:
            if verbose:
                print(f"  ✓ Successfully added face: {face_name} (ID: {result['id']})")
            return True, result['id'], face_name
        else:
            print(f"  ✗ No face detected in: {image_path.name}")
            return False, None, face_name
            
    except Exception as e:
        print(f"  ✗ Error processing {image_path.name}: {str(e)}")
        return False, None, face_name


async def main():
    parser = argparse.ArgumentParser(description='Register images from directory to face database')
    parser.add_argument('directory', help='Directory containing face images')
    parser.add_argument('--db-path', default='faces.db', help='Path to the database file (default: faces.db)')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of images to process at once (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed processing information')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without actually adding to database')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    
    # Get image files
    image_files = get_image_files(args.directory)
    
    if not image_files:
        print(f"No image files found in '{args.directory}'. Supported formats: jpg, jpeg, png, bmp, tiff, tif, webp")
        sys.exit(0)
    
    print(f"Found {len(image_files)} image files in '{args.directory}'")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No changes will be made to the database")
        print("The following images would be processed:")
        for img_file in image_files:
            print(f"  - {img_file.name} -> '{img_file.stem}'")
        return
    
    # Initialize processing module
    try:
        processing = await get_processing()
        await processing.start()
    except Exception as e:
        print(f"Error initializing processing module: {e}")
        print("Make sure model files are available in the models directory.")
        sys.exit(1)
    
    # Initialize face manager
    face_manager = FaceManager(db_path=args.db_path, processing=processing)
    
    # Process images in batches
    successful = 0
    failed = 0
    
    for i in range(0, len(image_files), args.batch_size):
        batch = image_files[i:i + args.batch_size]
        
        if args.verbose:
            print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(image_files)-1)//args.batch_size + 1}")
        
        # Process batch concurrently
        tasks = [process_image_file(img_file, face_manager, args.verbose) for img_file in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count results
        for result in results:
            if isinstance(result, Exception):
                print(f"  ✗ Exception during processing: {result}")
                failed += 1
            elif result[0]:  # Success
                successful += 1
            else:  # Failed to detect face
                failed += 1
    
    print(f"\nProcessing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")
    
    # Print summary of registered faces
    all_faces = face_manager.get_all_faces()
    print(f"\nTotal faces in database: {len(all_faces)}")


if __name__ == "__main__":
    asyncio.run(main())