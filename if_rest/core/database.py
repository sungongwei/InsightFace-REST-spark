import sqlite3
import os
from typing import List, Optional, Tuple
import numpy as np
import json
from datetime import datetime


import os


class FaceDatabase:
    """
    SQLite database for face management with CRUD operations.
    """

    def __init__(self, db_path: str = None):
        # Use environment variable if provided, otherwise default to faces.db
        self.db_path = db_path or os.getenv('DB_PATH', 'faces.db')

        # Ensure the directory for the database file exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:  # Only create directory if path contains a directory part
            os.makedirs(db_dir, exist_ok=True)

        self.init_db()
    
    def init_db(self):
        """Initialize the database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_data BLOB,
                gender INTEGER,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index on name for faster searches
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_faces_name ON faces(name)')
        
        conn.commit()
        conn.close()
    
    def add_face(self, name: str, embedding: np.ndarray, image_data: Optional[bytes] = None, 
                 gender: Optional[int] = None, age: Optional[int] = None) -> int:
        """Add a new face to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        cursor.execute('''
            INSERT INTO faces (name, embedding, image_data, gender, age)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, embedding_bytes, image_data, gender, age))
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def get_face_by_id(self, face_id: int) -> Optional[dict]:
        """Get a face by its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
        row = cursor.fetchone()
        
        if row:
            face = {
                'id': row[0],
                'name': row[1],
                'embedding': np.frombuffer(row[2], dtype=np.float32),
                'image_data': row[3],
                'gender': row[4],
                'age': row[5],
                'created_at': row[6],
                'updated_at': row[7]
            }
        else:
            face = None
        
        conn.close()
        return face
    
    def get_faces_by_name(self, name: str) -> List[dict]:
        """Get all faces with a specific name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces WHERE name LIKE ?', (f'%{name}%',))
        rows = cursor.fetchall()
        
        faces = []
        for row in rows:
            face = {
                'id': row[0],
                'name': row[1],
                'embedding': np.frombuffer(row[2], dtype=np.float32),
                'image_data': row[3],
                'gender': row[4],
                'age': row[5],
                'created_at': row[6],
                'updated_at': row[7]
            }
            faces.append(face)
        
        conn.close()
        return faces
    
    def get_all_faces(self) -> List[dict]:
        """Get all faces from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM faces')
        rows = cursor.fetchall()
        
        faces = []
        for row in rows:
            face = {
                'id': row[0],
                'name': row[1],
                'embedding': np.frombuffer(row[2], dtype=np.float32),
                'image_data': row[3],
                'gender': row[4],
                'age': row[5],
                'created_at': row[6],
                'updated_at': row[7]
            }
            faces.append(face)
        
        conn.close()
        return faces
    
    def update_face(self, face_id: int, name: Optional[str] = None, 
                    embedding: Optional[np.ndarray] = None, 
                    image_data: Optional[bytes] = None,
                    gender: Optional[int] = None, 
                    age: Optional[int] = None) -> bool:
        """Update an existing face."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        updates = []
        params = []
        
        if name is not None:
            updates.append('name = ?')
            params.append(name)
        
        if embedding is not None:
            updates.append('embedding = ?')
            params.append(embedding.tobytes())
        
        if image_data is not None:
            updates.append('image_data = ?')
            params.append(image_data)
        
        if gender is not None:
            updates.append('gender = ?')
            params.append(gender)
        
        if age is not None:
            updates.append('age = ?')
            params.append(age)
        
        if updates:
            updates.append('updated_at = ?')
            params.append(datetime.now().isoformat())
            
            query = f"UPDATE faces SET {', '.join(updates)} WHERE id = ?"
            params.append(face_id)
            
            cursor.execute(query, params)
            conn.commit()
            updated = cursor.rowcount > 0
        else:
            updated = False
        
        conn.close()
        return updated
    
    def delete_face(self, face_id: int) -> bool:
        """Delete a face by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        
        conn.close()
        return deleted
    
    def delete_faces_by_name(self, name: str) -> int:
        """Delete all faces with a specific name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
        conn.commit()
        deleted_count = cursor.rowcount
        
        conn.close()
        return deleted_count