import json
import os
import pickle
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import time

import numpy as np

from facefusion import state_manager, logger
from facefusion.face_store import FACE_STORE, create_frame_hash
from facefusion.face_selector import find_similar_faces, calc_face_distance
from facefusion.filesystem import is_video
from facefusion.typing import Face, VisionFrame, FaceSet
from facefusion.vision import get_video_frame, count_video_frame_total, detect_video_fps


class VideoFaceIndex:
    """
    Enhanced video face indexing system that builds upon existing face_store.py
    Provides persistent, efficient face caching for entire videos
    """
    
    def __init__(self, cache_dir: str = ".cache/face_index"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_lock = threading.Lock()
        self._memory_cache = {}
        self.current_video_path = None  # Track current video for context
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for face indexing metadata"""
        db_path = self.cache_dir / "face_index.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_index (
                    video_path TEXT PRIMARY KEY,
                    video_hash TEXT NOT NULL,
                    total_frames INTEGER NOT NULL,
                    fps REAL NOT NULL,
                    indexed_frames INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cache_file TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frame_faces (
                    video_path TEXT,
                    frame_number INTEGER,
                    face_index INTEGER,
                    face_hash TEXT,
                    bounding_box TEXT,
                    confidence REAL,
                    embedding_hash TEXT,
                    gender TEXT,
                    age_range TEXT,
                    PRIMARY KEY (video_path, frame_number, face_index),
                    FOREIGN KEY (video_path) REFERENCES video_index (video_path)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_matches (
                    video_path TEXT,
                    frame_number INTEGER,
                    match_key TEXT,  -- Hash of matching parameters (faces, references, distance)
                    similar_faces_data TEXT,  -- JSON serialized similar faces result
                    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (video_path, frame_number, match_key),
                    FOREIGN KEY (video_path) REFERENCES video_index (video_path)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ignored_faces (
                    video_path TEXT,
                    frame_number INTEGER,
                    face_index INTEGER,
                    face_hash TEXT,
                    ignored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (video_path, frame_number, face_index),
                    FOREIGN KEY (video_path) REFERENCES video_index (video_path)
                )
            """)
            conn.commit()
    
    def get_video_hash(self, video_path: str) -> str:
        """Generate hash for video file (based on path, size, modified time)"""
        if not os.path.exists(video_path):
            return ""
        
        stat = os.stat(video_path)
        content = f"{video_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def is_video_indexed(self, video_path: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if video is already indexed and get metadata"""
        video_hash = self.get_video_hash(video_path)
        
        with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
            cursor = conn.execute("""
                SELECT video_hash, total_frames, fps, indexed_frames, cache_file
                FROM video_index WHERE video_path = ?
            """, (video_path,))
            row = cursor.fetchone()
            
            if row and row[0] == video_hash:
                return True, {
                    'total_frames': row[1],
                    'fps': row[2], 
                    'indexed_frames': row[3],
                    'cache_file': row[4]
                }
            
            return False, {}
    
    def has_video_index(self, video_path: str) -> bool:
        """Simple check if video has cache index"""
        is_indexed, _ = self.is_video_indexed(video_path)
        return is_indexed
    
    def get_video_cache_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed cache information for video"""
        is_indexed, metadata = self.is_video_indexed(video_path)
        
        if not is_indexed:
            return {}
        
        # Get additional info like total faces and creation date
        with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
            # Count total faces
            cursor = conn.execute("""
                SELECT COUNT(*) FROM frame_faces WHERE video_path = ?
            """, (video_path,))
            total_faces = cursor.fetchone()[0]
            
            # Get creation date
            cursor = conn.execute("""
                SELECT last_updated FROM video_index WHERE video_path = ?
            """, (video_path,))
            created_at = cursor.fetchone()[0]
        
        return {
            'total_frames': metadata.get('total_frames', 0),
            'indexed_frames': metadata.get('indexed_frames', 0),
            'fps': metadata.get('fps', 0),
            'total_faces': total_faces,
            'created_at': created_at,
            'cache_file': metadata.get('cache_file', '')
        }
    
    def index_video_faces(self, video_path: str, progress_callback=None) -> bool:
        """
        Index all faces in a video, extending existing face detection
        Integrates with existing face_analyser.get_many_faces
        """
        if not is_video(video_path):
            logger.error(f"Invalid video path: {video_path}", __name__)
            return False
        
        # Check if already indexed
        is_indexed, metadata = self.is_video_indexed(video_path)
        if is_indexed and metadata.get('indexed_frames', 0) > 0:
            logger.info(f"Video already indexed: {video_path}", __name__)
            return True
        
        logger.info(f"Starting video face indexing for: {video_path}", __name__)
        
        try:
            from facefusion.face_analyser import get_many_faces  # Import here to avoid circular imports
            
            total_frames = count_video_frame_total(video_path)
            fps = detect_video_fps(video_path)
            video_hash = self.get_video_hash(video_path)
            
            # Create cache file for face data
            cache_filename = f"{video_hash}_faces.pkl"
            cache_file_path = self.cache_dir / cache_filename
            
            face_data = {}
            indexed_frames = 0
            
            # Process frames in batches for efficiency
            batch_size = 50
            for start_frame in range(0, total_frames, batch_size):
                end_frame = min(start_frame + batch_size, total_frames)
                
                # Process batch of frames
                batch_faces = {}
                for frame_num in range(start_frame, end_frame):
                    try:
                        vision_frame = get_video_frame(video_path, frame_num)
                        if vision_frame is not None:
                            # Use existing face detection system
                            faces = get_many_faces([vision_frame])
                            if faces:
                                batch_faces[frame_num] = self._serialize_faces(faces)
                                indexed_frames += 1
                                
                                # Store in database
                                self._store_frame_faces(video_path, frame_num, faces)
                        
                        if progress_callback:
                            progress = (frame_num + 1) / total_frames
                            progress_callback(progress)
                            
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_num}: {e}", __name__)
                        continue
                
                # Save batch to cache file
                face_data.update(batch_faces)
                
                # Periodically save progress
                if start_frame % (batch_size * 10) == 0:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(face_data, f)
            
            # Final save
            with open(cache_file_path, 'wb') as f:
                pickle.dump(face_data, f)
            
            # Update database
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO video_index 
                    (video_path, video_hash, total_frames, fps, indexed_frames, cache_file)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (video_path, video_hash, total_frames, fps, indexed_frames, cache_filename))
                conn.commit()
            
            logger.info(f"Video indexing complete: {indexed_frames}/{total_frames} frames", __name__)
            return True
            
        except Exception as e:
            logger.error(f"Error indexing video: {e}", __name__)
            return False
    
    def _serialize_faces(self, faces: List[Face]) -> List[Dict]:
        """Convert Face objects to serializable format"""
        serialized = []
        for face in faces:
            serialized.append({
                'bounding_box': face.bounding_box.tolist(),
                'confidence': float(face.score_set.get('detector', 0)),
                'embedding': face.embedding.tolist(),
                'normed_embedding': face.normed_embedding.tolist(),
                'gender': face.gender,
                'age': list(face.age) if hasattr(face.age, '__iter__') else face.age,
                'race': face.race,
                'landmarks': {
                    '5': face.landmark_set.get('5').tolist() if face.landmark_set.get('5') is not None else None,
                    '68': face.landmark_set.get('68').tolist() if face.landmark_set.get('68') is not None else None
                }
            })
        return serialized
    
    def _store_frame_faces(self, video_path: str, frame_number: int, faces: List[Face]):
        """Store face data in database"""
        with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
            # Clear existing faces for this frame
            conn.execute("""
                DELETE FROM frame_faces 
                WHERE video_path = ? AND frame_number = ?
            """, (video_path, frame_number))
            
            # Insert new faces
            for face_idx, face in enumerate(faces):
                embedding_hash = hashlib.sha256(face.embedding.tobytes()).hexdigest()[:16]
                
                conn.execute("""
                    INSERT INTO frame_faces 
                    (video_path, frame_number, face_index, face_hash, bounding_box, 
                     confidence, embedding_hash, gender, age_range, race)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_path, frame_number, face_idx,
                    f"{frame_number}_{face_idx}",  # Simple hash for now
                    json.dumps(face.bounding_box.tolist()),
                    float(face.score_set.get('detector', 0)),
                    embedding_hash,
                    face.gender,
                    str(face.age),
                    face.race
                ))
            conn.commit()
    
    def get_cached_faces(self, video_path: str, frame_number: int) -> Optional[List[Face]]:
        """Retrieve cached faces for a specific frame"""
        is_indexed, metadata = self.is_video_indexed(video_path)
        if not is_indexed:
            return None
        
        cache_file = self.cache_dir / metadata['cache_file']
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                face_data = pickle.load(f)
            
            if frame_number in face_data:
                return self._deserialize_faces(face_data[frame_number])
            
        except Exception as e:
            logger.error(f"Error reading cached faces: {e}", __name__)
        
        return None
    
    def get_all_frame_faces(self) -> Dict[int, List[Face]]:
        """Retrieve all frame faces from the current video's cache"""
        if not self.current_video_path:
            return {}
        
        is_indexed, metadata = self.is_video_indexed(self.current_video_path)
        if not is_indexed:
            return {}

        cache_file = self.cache_dir / metadata['cache_file']
        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, 'rb') as f:
                face_data = pickle.load(f)

            # Convert all serialized faces back to Face objects
            result = {}
            for frame_number, frame_face_data in face_data.items():
                result[frame_number] = self._deserialize_faces(frame_face_data)

            return result

        except Exception as e:
            logger.error(f"Error reading all cached faces: {e}", __name__)
            return {}
    
    def get_cached_face_matches(self, video_path: str, frame_number: int, 
                               faces: List[Face], reference_faces: FaceSet, 
                               face_distance: float) -> Optional[List[Face]]:
        """Retrieve cached face matching results"""
        try:
            # Create a cache key based on the matching parameters
            match_key = self._create_match_key(faces, reference_faces, face_distance)
            
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                cursor = conn.execute("""
                    SELECT similar_faces_data FROM face_matches 
                    WHERE video_path = ? AND frame_number = ? AND match_key = ?
                """, (video_path, frame_number, match_key))
                row = cursor.fetchone()
                
                if row:
                    # Deserialize the cached similar faces
                    similar_faces_data = json.loads(row[0])
                    return self._deserialize_faces(similar_faces_data)
                    
        except Exception as e:
            logger.debug(f"Error retrieving cached face matches: {e}", __name__)
        
        return None
    
    def cache_face_matches(self, video_path: str, frame_number: int,
                          faces: List[Face], reference_faces: FaceSet,
                          face_distance: float, similar_faces: List[Face]) -> None:
        """Cache face matching results for future use"""
        try:
            match_key = self._create_match_key(faces, reference_faces, face_distance)
            similar_faces_data = json.dumps(self._serialize_faces(similar_faces))
            
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO face_matches 
                    (video_path, frame_number, match_key, similar_faces_data, last_checked)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (video_path, frame_number, match_key, similar_faces_data))
                conn.commit()
                
        except Exception as e:
            logger.debug(f"Error caching face matches: {e}", __name__)
    
    def _create_match_key(self, faces: List[Face], reference_faces: FaceSet, 
                         face_distance: float) -> str:
        """Create a unique key for face matching parameters"""
        # Create hash based on face embeddings and distance threshold
        face_hashes = []
        for face in faces:
            face_hash = hashlib.sha256(face.embedding.tobytes()).hexdigest()[:8]
            face_hashes.append(face_hash)
        
        ref_hashes = []
        for ref_face in reference_faces:
            ref_hash = hashlib.sha256(ref_face.embedding.tobytes()).hexdigest()[:8]
            ref_hashes.append(ref_hash)
        
        # Combine all parameters into a single key
        key_data = f"faces:{','.join(sorted(face_hashes))}_refs:{','.join(sorted(ref_hashes))}_dist:{face_distance:.3f}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def _deserialize_faces(self, face_data: List[Dict]) -> List[Face]:
        """Convert serialized face data back to Face objects"""
        from facefusion.typing import Face  # Import here to avoid circular imports
        
        faces = []
        for data in face_data:
            # Reconstruct Face namedtuple
            face = Face(
                bounding_box=np.array(data['bounding_box']),
                score_set={'detector': data['confidence']},
                landmark_set={
                    '5': np.array(data['landmarks']['5']) if data['landmarks']['5'] else None,
                    '68': np.array(data['landmarks']['68']) if data['landmarks']['68'] else None
                },
                angle=0,  # Not stored in current serialization
                embedding=np.array(data['embedding']),
                normed_embedding=np.array(data['normed_embedding']),
                gender=data['gender'],
                age=data['age'],
                race=data['race']
            )
            faces.append(face)
        
        return faces
    
    def find_unmatched_faces(self, video_path: str, reference_faces: Dict[int, List[Face]], 
                           face_distance_threshold: float = 0.6) -> Dict[int, List[Face]]:
        """
        Find all unmatched faces in the video
        Returns: {frame_number: [face, ...]}
        """
        unmatched_faces = {}
        
        is_indexed, metadata = self.is_video_indexed(video_path)
        if not is_indexed:
            logger.warning(f"Video not indexed: {video_path}", __name__)
            return unmatched_faces
        
        cache_file = self.cache_dir / metadata['cache_file']
        if not cache_file.exists():
            logger.warning(f"Cache file not found: {cache_file}", __name__)
            return unmatched_faces
        
        try:
            with open(cache_file, 'rb') as f:
                face_data = pickle.load(f)
            
            # Flatten reference faces to a single list for comparison
            reference_face_list = []
            for face_list in reference_faces.values():
                reference_face_list.extend(face_list)
            
            if not reference_face_list:
                logger.warning("No reference faces provided for matching", __name__)
                return unmatched_faces
            
            for frame_number, frame_faces in face_data.items():
                frame_unmatched = []
                
                for face_idx, face_data_item in enumerate(frame_faces):
                    face = self._deserialize_faces([face_data_item])[0]
                    
                    # Check if face matches any reference face
                    is_matched = False
                    for reference_face in reference_face_list:
                        distance = calc_face_distance(face, reference_face)
                        if distance < face_distance_threshold:
                            is_matched = True
                            break
                    
                    if not is_matched:
                        frame_unmatched.append(face)
                
                if frame_unmatched:
                    unmatched_faces[frame_number] = frame_unmatched
            
            logger.info(f"Found unmatched faces in {len(unmatched_faces)} frames", __name__)
            
        except Exception as e:
            logger.error(f"Error finding unmatched faces: {e}", __name__)
        
        return unmatched_faces
    
    def clear_video_index(self, video_path: str = None):
        """Clear video index cache (all videos or specific video)"""
        with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
            if video_path:
                # Get cache file before deleting
                cursor = conn.execute("""
                    SELECT cache_file FROM video_index WHERE video_path = ?
                """, (video_path,))
                row = cursor.fetchone()
                
                if row:
                    cache_file = self.cache_dir / row[0]
                    if cache_file.exists():
                        cache_file.unlink()
                
                # Delete from database
                conn.execute("DELETE FROM ignored_faces WHERE video_path = ?", (video_path,))
                conn.execute("DELETE FROM face_matches WHERE video_path = ?", (video_path,))
                conn.execute("DELETE FROM frame_faces WHERE video_path = ?", (video_path,))
                conn.execute("DELETE FROM video_index WHERE video_path = ?", (video_path,))
            else:
                # Clear all
                conn.execute("DELETE FROM ignored_faces")
                conn.execute("DELETE FROM face_matches")
                conn.execute("DELETE FROM frame_faces") 
                conn.execute("DELETE FROM video_index")
                
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*_faces.pkl"):
                    cache_file.unlink()
            
            conn.commit()
    
    def mark_face_as_ignored(self, video_path: str, frame_number: int, face_index: int, face: Face) -> None:
        """Mark a specific face as ignored"""
        try:
            face_hash = hashlib.sha256(face.embedding.tobytes()).hexdigest()[:16]
            
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO ignored_faces 
                    (video_path, frame_number, face_index, face_hash, ignored_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (video_path, frame_number, face_index, face_hash))
                conn.commit()
                
            logger.info(f"Marked face as ignored: frame {frame_number}, face {face_index}", __name__)
            
        except Exception as e:
            logger.error(f"Error marking face as ignored: {e}", __name__)
    
    def is_face_ignored(self, video_path: str, frame_number: int, face_index: int) -> bool:
        """Check if a specific face is marked as ignored"""
        try:
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                cursor = conn.execute("""
                    SELECT 1 FROM ignored_faces 
                    WHERE video_path = ? AND frame_number = ? AND face_index = ?
                """, (video_path, frame_number, face_index))
                return cursor.fetchone() is not None
                
        except Exception as e:
            logger.debug(f"Error checking if face is ignored: {e}", __name__)
            return False
    
    def get_ignored_faces(self, video_path: str) -> Dict[int, List[int]]:
        """Get all ignored faces for a video. Returns {frame_number: [face_indices]}"""
        ignored_faces = {}
        
        try:
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                cursor = conn.execute("""
                    SELECT frame_number, face_index FROM ignored_faces 
                    WHERE video_path = ?
                    ORDER BY frame_number, face_index
                """, (video_path,))
                
                for frame_number, face_index in cursor.fetchall():
                    if frame_number not in ignored_faces:
                        ignored_faces[frame_number] = []
                    ignored_faces[frame_number].append(face_index)
                    
        except Exception as e:
            logger.debug(f"Error getting ignored faces: {e}", __name__)
        
        return ignored_faces
    
    def filter_ignored_faces(self, video_path: str, frame_number: int, faces: List[Face]) -> List[Face]:
        """Filter out ignored faces from a list of faces"""
        if not faces:
            return faces
        
        try:
            ignored_indices = set()
            with sqlite3.connect(str(self.cache_dir / "face_index.db")) as conn:
                cursor = conn.execute("""
                    SELECT face_index FROM ignored_faces 
                    WHERE video_path = ? AND frame_number = ?
                """, (video_path, frame_number))
                
                for (face_index,) in cursor.fetchall():
                    ignored_indices.add(face_index)
            
            # Filter out ignored faces
            filtered_faces = []
            for i, face in enumerate(faces):
                if i not in ignored_indices:
                    filtered_faces.append(face)
            
            if len(filtered_faces) != len(faces):
                logger.debug(f"Filtered out {len(faces) - len(filtered_faces)} ignored faces from frame {frame_number}", __name__)
            
            return filtered_faces
            
        except Exception as e:
            logger.debug(f"Error filtering ignored faces: {e}", __name__)
            return faces


# Global instance
VIDEO_FACE_INDEX = VideoFaceIndex() 