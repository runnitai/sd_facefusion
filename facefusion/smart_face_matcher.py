import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import cv2

from facefusion import logger
from facefusion.face_selector import calc_face_distance, find_similar_faces
from facefusion.typing import Face, BoundingBox, FaceSet
from facefusion.video_face_index import VIDEO_FACE_INDEX
from facefusion import state_manager


class SmartFaceMatcher:
    """
    Smart face matching system that uses temporal consistency and position analysis
    to automatically detect and match faces across video frames
    """
    
    def __init__(self):
        self.position_threshold = 0.3  # normalized distance threshold for position similarity
        self.temporal_window = 5  # frames to look back/forward
        self.confidence_threshold = 0.7
        self.position_weight = 0.3
        self.embedding_weight = 0.7
    
    def calculate_position_similarity(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate similarity based on bounding box position and size"""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        
        # Calculate center distance
        center_distance = np.linalg.norm(center1 - center2)
        
        # Calculate size similarity  
        size1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        size2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
        
        # Combine distance and size similarity
        position_similarity = max(0, 1 - (center_distance / self.position_threshold)) * size_ratio
        
        return position_similarity
    
    def get_temporal_context(self, video_path: str, frame_number: int, 
                           window_size: int = None) -> Dict[int, List[Face]]:
        """Get faces from surrounding frames for temporal context"""
        if window_size is None:
            window_size = self.temporal_window
        
        context_faces = {}
        start_frame = max(0, frame_number - window_size // 2)
        end_frame = frame_number + window_size // 2
        
        for frame_num in range(start_frame, end_frame + 1):
            if frame_num == frame_number:
                continue  # Skip current frame
                
            cached_faces = VIDEO_FACE_INDEX.get_cached_faces(video_path, frame_num)
            if cached_faces:
                context_faces[frame_num] = cached_faces
        
        return context_faces
    
    def find_consistent_faces(self, video_path: str, target_frame: int, 
                            target_faces: List[Face]) -> Dict[int, List[Tuple[Face, float]]]:
        """
        Find faces that appear consistently in similar positions across frames
        Returns: {face_index: [(matched_face, confidence), ...]}
        """
        temporal_context = self.get_temporal_context(video_path, target_frame)
        consistent_matches = {}
        
        for target_idx, target_face in enumerate(target_faces):
            matches = []
            
            for context_frame, context_faces in temporal_context.items():
                frame_distance = abs(context_frame - target_frame)
                temporal_weight = max(0, 1 - frame_distance / self.temporal_window)
                
                for context_face in context_faces:
                    # Calculate embedding similarity
                    embedding_similarity = 1 - calc_face_distance(target_face, context_face)
                    
                    # Calculate position similarity
                    position_similarity = self.calculate_position_similarity(
                        target_face.bounding_box, context_face.bounding_box
                    )
                    
                    # Combined confidence score
                    combined_score = (
                        self.embedding_weight * embedding_similarity +
                        self.position_weight * position_similarity
                    ) * temporal_weight
                    
                    if combined_score > self.confidence_threshold:
                        matches.append((context_face, combined_score))
            
            if matches:
                # Sort by confidence and keep the best matches
                matches.sort(key=lambda x: x[1], reverse=True)
                consistent_matches[target_idx] = matches[:5]  # Keep top 5
        
        return consistent_matches
    
    def auto_match_unmatched_faces(self, video_path: str, 
                                 unmatched_faces: Dict[int, List[Tuple[int, Face]]],
                                 source_faces: Dict[int, Face]) -> Dict[int, Dict[int, str]]:
        """
        Automatically match unmatched faces using temporal consistency
        Returns: {frame_number: {face_index: 'auto_matched'}}
        """
        auto_matches = {}
        source_face_list = list(source_faces.values())
        
        logger.info(f"Starting auto-matching for {len(unmatched_faces)} frames", __name__)
        
        for frame_number, frame_unmatched in unmatched_faces.items():
            if frame_number not in auto_matches:
                auto_matches[frame_number] = {}
            
            # Get all faces in current frame (for context)
            all_frame_faces = VIDEO_FACE_INDEX.get_cached_faces(video_path, frame_number)
            if not all_frame_faces:
                continue
            
            # Find consistent faces around this frame
            consistent_matches = self.find_consistent_faces(video_path, frame_number, all_frame_faces)
            
            for face_idx, face in frame_unmatched:
                # Check if this face appears consistently in similar positions
                if face_idx in consistent_matches:
                    context_matches = consistent_matches[face_idx]
                    
                    # Check if context matches align with any source face
                    best_source_match = None
                    best_confidence = 0
                    
                    for source_idx, source_face in source_faces.items():
                        total_confidence = 0
                        match_count = 0
                        
                        for matched_face, confidence in context_matches:
                            context_source_distance = calc_face_distance(matched_face, source_face)
                            if context_source_distance < 0.6:  # Reasonable match threshold
                                total_confidence += confidence * (1 - context_source_distance)
                                match_count += 1
                        
                        if match_count > 0:
                            avg_confidence = total_confidence / match_count
                            if avg_confidence > best_confidence:
                                best_confidence = avg_confidence
                                best_source_match = source_idx
                    
                    # If we found a consistent match, mark it as auto-matched
                    if best_source_match is not None and best_confidence > 0.5:
                        auto_matches[frame_number][face_idx] = 'auto_matched'
                        logger.debug(f"Auto-matched face {face_idx} in frame {frame_number} "
                                   f"to source {best_source_match} (confidence: {best_confidence:.3f})", __name__)
        
        matched_frames = len([f for f in auto_matches.values() if f])
        logger.info(f"Auto-matching complete: {matched_frames} frames with auto-matches", __name__)
        
        return auto_matches
    
    def detect_tracking_failures(self, video_path: str, source_faces: Dict[int, Face],
                                consecutive_threshold: int = 5) -> List[Tuple[int, int, str]]:
        """
        Detect potential face tracking failures (sudden appearance/disappearance)
        Returns: [(start_frame, end_frame, failure_type), ...]
        """
        failures = []
        
        is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(video_path)
        if not is_indexed:
            return failures
        
        # Track presence of each source face across frames
        total_frames = metadata['total_frames']
        source_tracking = {src_idx: [] for src_idx in source_faces.keys()}
        
        for frame_num in range(total_frames):
            cached_faces = VIDEO_FACE_INDEX.get_cached_faces(video_path, frame_num)
            
            if cached_faces:
                for src_idx, source_face in source_faces.items():
                    # Check if source face is present in this frame
                    is_present = False
                    for face in cached_faces:
                        distance = calc_face_distance(face, source_face)
                        if distance < 0.6:  # Match threshold
                            is_present = True
                            break
                    
                    source_tracking[src_idx].append(is_present)
            else:
                # No faces detected in frame
                for src_idx in source_faces.keys():
                    source_tracking[src_idx].append(False)
        
        # Detect consecutive gaps (potential tracking failures)
        for src_idx, presence_list in source_tracking.items():
            current_gap_start = None
            gap_length = 0
            
            for frame_idx, is_present in enumerate(presence_list):
                if not is_present:
                    if current_gap_start is None:
                        current_gap_start = frame_idx
                    gap_length += 1
                else:
                    if gap_length >= consecutive_threshold:
                        failures.append((
                            current_gap_start, 
                            frame_idx - 1, 
                            f'source_{src_idx}_gap'
                        ))
                    current_gap_start = None
                    gap_length = 0
            
            # Check if gap extends to end of video
            if gap_length >= consecutive_threshold:
                failures.append((
                    current_gap_start, 
                    len(presence_list) - 1, 
                    f'source_{src_idx}_gap'
                ))
        
        logger.info(f"Detected {len(failures)} potential tracking failures", __name__)
        return failures
    
    def suggest_manual_review_frames(self, video_path: str, 
                                   source_faces: Dict[int, Face]) -> List[int]:
        """
        Suggest frames that might need manual review based on various factors
        """
        review_frames = []
        
        # 1. Frames with tracking failures
        failures = self.detect_tracking_failures(video_path, source_faces)
        for start_frame, end_frame, _ in failures:
            # Suggest frames around the failure boundaries
            review_frames.extend([start_frame, end_frame, (start_frame + end_frame) // 2])
        
        # 2. Frames with many unmatched faces
        unmatched = VIDEO_FACE_INDEX.find_unmatched_faces(video_path, source_faces)
        for frame_num, unmatched_faces in unmatched.items():
            if len(unmatched_faces) > 2:  # Many unmatched faces
                review_frames.append(frame_num)
        
        # 3. Frames with low detection confidence
        is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(video_path)
        if is_indexed:
            total_frames = metadata['total_frames']
            for frame_num in range(0, total_frames, 100):  # Sample every 100 frames
                cached_faces = VIDEO_FACE_INDEX.get_cached_faces(video_path, frame_num)
                if cached_faces:
                    avg_confidence = np.mean([
                        face.score_set.get('detector', 0) for face in cached_faces
                    ])
                    if avg_confidence < 0.5:  # Low confidence
                        review_frames.append(frame_num)
        
        # Remove duplicates and sort
        review_frames = sorted(list(set(review_frames)))
        
        logger.info(f"Suggested {len(review_frames)} frames for manual review", __name__)
        return review_frames

    def find_unmatched_faces(self, frame_faces: List[Face], reference_faces: Dict[int, FaceSet], 
                           source_faces: Dict[int, Face]) -> List[Face]:
        """
        Find faces in the frame that don't match any reference or source faces.
        Uses the exact same matching logic as face_swapper.
        """
        if not frame_faces:
            return []
        
        # Get the face distance threshold used by face_swapper
        face_distance_threshold = state_manager.get_item('reference_face_distance') or 0.6
        
        # Combine reference and source face keys like face_swapper does
        reference_face_keys = set(reference_faces.keys())
        source_face_keys = set(source_faces.keys())
        all_keys = reference_face_keys.union(source_face_keys)
        
        unmatched_faces = []
        
        for face in frame_faces:
            is_matched = False
            
            # Check against all reference faces using the same logic as face_swapper
            for face_idx in all_keys:
                ref_faces = reference_faces.get(face_idx)
                if ref_faces:
                    # Use the exact same find_similar_faces function as face_swapper
                    similar_faces = find_similar_faces([face], ref_faces, face_distance_threshold)
                    if similar_faces:
                        is_matched = True
                        break
            
            if not is_matched:
                unmatched_faces.append(face)
        
        return unmatched_faces
    
    def auto_match_face(self, target_face: Face, reference_faces: Dict[int, FaceSet], 
                       frame_history: List[Dict]) -> Optional[int]:
        """
        Automatically match a face using temporal consistency and position similarity.
        Uses the same distance calculation as face_swapper.
        """
        if not reference_faces:
            return None
        
        face_distance_threshold = state_manager.get_item('reference_face_distance') or 0.6
        best_match_idx = None
        best_distance = float('inf')
        
        # First try direct embedding similarity like face_swapper
        for face_idx, ref_faces in reference_faces.items():
            similar_faces = find_similar_faces([target_face], ref_faces, face_distance_threshold)
            if similar_faces:
                # Calculate the best distance to this reference set
                min_distance = min(calc_face_distance(target_face, ref_face) for ref_face in ref_faces)
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_match_idx = face_idx
        
        # If we found a match based on embedding similarity, return it
        if best_match_idx is not None:
            return best_match_idx
        
        # If no embedding match, try temporal consistency with position similarity
        return self._find_temporal_match(target_face, reference_faces, frame_history, face_distance_threshold)
    
    def _find_temporal_match(self, target_face: Face, reference_faces: Dict[int, FaceSet], 
                           frame_history: List[Dict], face_distance_threshold: float) -> Optional[int]:
        """Find match using temporal consistency and position similarity"""
        if not frame_history:
            return None
        
        # Look at recent frames for temporal consistency
        recent_frames = frame_history[-self.temporal_window:]
        position_votes = {}
        
        for frame_data in recent_frames:
            frame_faces = frame_data.get('faces', [])
            for face in frame_faces:
                if self._is_similar_position(target_face, face):
                    # Check which reference this face matched to
                    for face_idx, ref_faces in reference_faces.items():
                        similar_faces = find_similar_faces([face], ref_faces, face_distance_threshold)
                        if similar_faces:
                            position_votes[face_idx] = position_votes.get(face_idx, 0) + 1
                            break
        
        # Return the face_idx with the most votes
        if position_votes:
            return max(position_votes.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _is_similar_position(self, face1: Face, face2: Face) -> bool:
        """Check if two faces are in similar positions"""
        if not (hasattr(face1, 'bounding_box') and hasattr(face2, 'bounding_box')):
            return False
        
        # Calculate normalized center positions
        center1 = self._get_face_center(face1)
        center2 = self._get_face_center(face2)
        
        # Calculate distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance < self.position_threshold
    
    def _get_face_center(self, face: Face) -> Tuple[float, float]:
        """Get normalized center position of a face"""
        bbox = face.bounding_box
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        return (center_x, center_y)
    
    def batch_match_faces(self, unmatched_faces: List[Face], reference_faces: Dict[int, FaceSet], 
                         frame_history: List[Dict]) -> Dict[Face, Optional[int]]:
        """Batch process multiple faces for matching"""
        matches = {}
        for face in unmatched_faces:
            match_idx = self.auto_match_face(face, reference_faces, frame_history)
            matches[face] = match_idx
        return matches


# Global instance
SMART_FACE_MATCHER = SmartFaceMatcher() 