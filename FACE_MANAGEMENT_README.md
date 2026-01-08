# Face Management and Search System

This system extends the InsightFace-REST API with face management and search capabilities.

## Features

- **Face Management**: Add, update, delete, and search faces in a SQLite database
- **Face Search**: Find similar faces using FAISS-based similarity search
- **Web Interfaces**: Simple web pages for face management and search
- **GPU Support**: Optional GPU acceleration for FAISS similarity search
- **Data Persistence**: Docker volumes for persistent storage of face data

## Installation

1. Install the base requirements:
```bash
pip install -r requirements.txt
```

2. Install FAISS based on your hardware:
   - For CPU-only systems:
     ```bash
     pip install faiss-cpu
     ```

   - For GPU-enabled systems (requires CUDA):
     ```bash
     pip install faiss-gpu
     ```

## Configuration

- To enable GPU support, set the environment variable:
  ```bash
  export USE_GPU=true
  ```

- To specify a custom database path:
  ```bash
  export DB_PATH=/path/to/your/faces.db
  ```

## Usage

1. Start the server:
```bash
python -m if_rest.api.main
```

2. Access the web interfaces:
   - Main dashboard: http://localhost:18080
   - Face management: http://localhost:18080/static/face_management.html
   - Face search: http://localhost:18080/static/face_search.html

3. API endpoints:
   - Face management: `/v1/faces`
   - Face search: `/v1/search`

## Docker Deployment with Persistent Storage

The system includes Docker Compose configurations with persistent storage for face data:

### Available Compose Files
- `docker-compose.yml` - Single GPU configuration with persistent storage
- `docker-compose-cpu.yml` - CPU-only configuration with persistent storage
- `docker-compose-multi-gpu.yml` - Multi-GPU configuration with shared persistent storage
- `docker-compose-v2.yml` - Modern compose with profiles and persistent storage

### Volume Configuration
All compose files include a named volume `face_data` mounted at `/app/data` in containers:
- The SQLite database is stored at `/app/data/faces.db`
- This ensures face data persists across container restarts
- Multiple containers can share the same face data when using multi-GPU setups

### Environment Variables in Docker
- `DB_PATH=/app/data/faces.db` - Path to the persistent database
- `USE_GPU=true/false` - Enable/disable GPU acceleration

### Running with Docker Compose
```bash
# For GPU version
docker-compose -f compose/docker-compose.yml up -d

# For CPU version
docker-compose -f compose/docker-compose-cpu.yml up -d

# Using compose v2 with profiles
docker-compose -f compose/docker-compose-v2.yml --profile gpu up -d
docker-compose -f compose/docker-compose-v2.yml --profile cpu up -d
```

## API Endpoints

### Face Management
- `POST /v1/faces` - Add a new face
- `GET /v1/faces/{id}` - Get face by ID
- `GET /v1/faces` - Get all faces
- `GET /v1/faces/search?name=` - Search faces by name
- `PUT /v1/faces/{id}` - Update face
- `DELETE /v1/faces/{id}` - Delete face by ID
- `DELETE /v1/faces?name=` - Delete all faces by name

### Face Search
- `POST /v1/search` - Search for similar faces using image
- `POST /v1/search/embedding` - Search for similar faces using embedding

## Database

The system uses SQLite to store:
- Face embeddings (512-dimensional vectors)
- Face metadata (name, gender, age)
- Associated images
- Creation/update timestamps

## FAISS Search

The system uses FAISS for efficient similarity search:
- Cosine similarity for face matching
- GPU acceleration support (optional)
- Index automatically updates when faces are added/removed

## Bulk Registration

The system includes scripts for bulk registration of faces from a directory:

### Local Register Faces Script
Use the `register_faces.py` script to register all images in a directory directly to the local database:

```bash
# Basic usage
python register_faces.py /path/to/images

# With custom database path
python register_faces.py /path/to/images --db-path /path/to/custom.db

# Verbose mode with batch processing
python register_faces.py /path/to/images --verbose --batch-size 5

# Dry run to see what would be processed
python register_faces.py /path/to/images --dry-run
```

### API Register Faces Script
Use the `register_faces_api.py` script to register all images via the API (for remote servers):

```bash
# Basic usage (registers to localhost:18080)
python register_faces_api.py /path/to/images

# Register to a remote server
python register_faces_api.py /path/to/images --api-url https://myserver.com

# Verbose mode with batch processing
python register_faces_api.py /path/to/images --verbose --batch-size 3

# Dry run to see what would be processed
python register_faces_api.py /path/to/images --dry-run
```

Both scripts will:
- Read all supported image files from the specified directory
- Use the filename (without extension) as the face name
- Extract face embeddings using the InsightFace model
- Store the faces in the database with associated metadata
- Provide detailed progress information
- Support batch processing for large datasets