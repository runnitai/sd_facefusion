"""
Face Cache System Initialization Module

This module ensures all face cache components are properly initialized
and integrated with the existing FaceFusion system.
"""

from facefusion import state_manager, logger


def init_face_cache_system():
    """Initialize the face cache system with default settings"""
    
    # Initialize state manager items for face cache
    default_settings = {
        'video_face_cache_enabled': True,
        'cache_unmatched_faces': True,
        'auto_match_faces': True,
        'face_cache_batch_size': 50,
        'smart_matching_enabled': True,
        'temporal_window_size': 10,
        'position_threshold': 50,
        'face_cache_directory': '.cache/face_index'
    }
    
    for key, default_value in default_settings.items():
        if state_manager.get_item(key) is None:
            state_manager.init_item(key, default_value)
    
    logger.info("Face cache system initialized", __name__)


def register_face_cache_args(parser):
    """Register face cache arguments with the main argument parser"""
    
    # Face cache group
    face_cache_group = parser.add_argument_group('face cache', 'Face cache and indexing options')
    
    face_cache_group.add_argument('--video-face-cache-enabled', 
                                 action='store_true', 
                                 default=True,
                                 help='Enable video face caching for improved performance')
    
    face_cache_group.add_argument('--disable-video-face-cache', 
                                 dest='video_face_cache_enabled',
                                 action='store_false',
                                 help='Disable video face caching')
    
    face_cache_group.add_argument('--cache-unmatched-faces', 
                                 action='store_true', 
                                 default=True,
                                 help='Cache unmatched faces for review')
    
    face_cache_group.add_argument('--auto-match-faces', 
                                 action='store_true', 
                                 default=True,
                                 help='Enable automatic face matching using temporal consistency')
    
    face_cache_group.add_argument('--face-cache-batch-size', 
                                 type=int, 
                                 default=50,
                                 help='Batch size for video face indexing (default: 50)')
    
    face_cache_group.add_argument('--face-cache-directory', 
                                 default='.cache/face_index',
                                 help='Directory for face cache storage (default: .cache/face_index)')
    
    face_cache_group.add_argument('--temporal-window-size', 
                                 type=int, 
                                 default=10,
                                 help='Temporal window size for smart matching (default: 10)')
    
    face_cache_group.add_argument('--position-threshold', 
                                 type=int, 
                                 default=50,
                                 help='Position threshold for face matching in pixels (default: 50)')


def check_face_cache_dependencies():
    """Check if all dependencies for face cache system are available"""
    missing_deps = []
    
    try:
        import sqlite3
    except ImportError:
        missing_deps.append('sqlite3')
    
    try:
        import pickle
    except ImportError:
        missing_deps.append('pickle')
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append('numpy')
    
    if missing_deps:
        logger.error(f"Missing dependencies for face cache system: {', '.join(missing_deps)}", __name__)
        return False
    
    return True


def cleanup_face_cache_system():
    """Cleanup face cache system resources"""
    try:
        from facefusion.video_face_index import VIDEO_FACE_INDEX
        # Any cleanup needed for the video face index
        logger.debug("Face cache system cleanup completed", __name__)
    except ImportError:
        pass


# Auto-initialize when module is imported
if check_face_cache_dependencies():
    init_face_cache_system()
else:
    logger.warning("Face cache system disabled due to missing dependencies", __name__) 