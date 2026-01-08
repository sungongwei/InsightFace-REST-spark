#!/usr/bin/env python3
"""
Script to register images from a directory to the face database via API.
The filename (without extension) will be used as the face name.
"""

import argparse
import os
import requests
import asyncio
import aiohttp
import sys
from pathlib import Path
import json
from typing import List, Tuple


async def get_image_files(directory: str) -> List[Path]:
    """Get all image files from the specified directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


async def register_face_via_api(session: aiohttp.ClientSession, api_url: str, image_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Register a single face via the API."""
    try:
        # Extract face name from filename (without extension)
        face_name = image_path.stem
        
        if verbose:
            print(f"Registering: {image_path.name} -> Face name: '{face_name}'")
        
        # Prepare the file for upload
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create form data
        data = aiohttp.FormData()
        data.add_field('name', face_name)
        data.add_field('file', image_data, filename=image_path.name, content_type='image/jpeg')
        
        # Make API request
        async with session.post(f"{api_url}/api/v1/faces", data=data) as response:
            if response.status == 200:
                result = await response.json()
                if verbose:
                    print(f"  ✓ Successfully registered face: {face_name} (ID: {result.get('id', 'unknown')})")
                return True, face_name
            else:
                error_text = await response.text()
                print(f"  ✗ Failed to register {image_path.name}: {response.status} - {error_text}")
                return False, face_name
                
    except Exception as e:
        print(f"  ✗ Error registering {image_path.name}: {str(e)}")
        return False, face_name


async def main():
    parser = argparse.ArgumentParser(description='Register images from directory to face database via API')
    parser.add_argument('directory', help='Directory containing face images')
    parser.add_argument('--api-url', default='http://localhost:18080', help='API base URL (default: http://localhost:18080)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of images to process at once (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed processing information')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without actually calling the API')
    
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        sys.exit(1)
    
    # Get image files
    image_files = await get_image_files(args.directory)
    
    if not image_files:
        print(f"No image files found in '{args.directory}'. Supported formats: jpg, jpeg, png, bmp, tiff, tif, webp")
        sys.exit(0)
    
    print(f"Found {len(image_files)} image files in '{args.directory}'")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No API calls will be made")
        print("The following images would be registered:")
        for img_file in image_files:
            print(f"  - {img_file.name} -> '{img_file.stem}'")
        return
    
    # Test API connection first
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.api_url}/docs") as response:
                if response.status != 200:
                    print(f"Warning: Cannot reach API at {args.api_url}. Status: {response.status}")
                    # Continue anyway as the endpoint might be different
    except Exception as e:
        print(f"Warning: Cannot connect to API at {args.api_url}: {e}")
        print("Attempting to continue...")
    
    # Process images in batches
    successful = 0
    failed = 0
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(image_files), args.batch_size):
            batch = image_files[i:i + args.batch_size]
            
            if args.verbose:
                print(f"\nProcessing batch {i//args.batch_size + 1}/{(len(image_files)-1)//args.batch_size + 1}")
            
            # Process batch concurrently
            tasks = [register_face_via_api(session, args.api_url, img_file, args.verbose) for img_file in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count results
            for result in results:
                if isinstance(result, Exception):
                    print(f"  ✗ Exception during registration: {result}")
                    failed += 1
                elif result[0]:  # Success
                    successful += 1
                else:  # Failed to register
                    failed += 1
    
    print(f"\nRegistration completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")


if __name__ == "__main__":
    asyncio.run(main())