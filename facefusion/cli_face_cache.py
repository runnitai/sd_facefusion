#!/usr/bin/env python3

import argparse
import sys
from typing import Optional

from facefusion import logger, state_manager
from facefusion.face_analyser import get_average_faces
from facefusion.filesystem import is_video
from facefusion.video_face_index import VIDEO_FACE_INDEX
from facefusion.smart_face_matcher import SMART_FACE_MATCHER


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI parser for face cache operations"""
    parser = argparse.ArgumentParser(
        prog='face_cache_cli',
        description='Command-line interface for FaceFusion face cache operations'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index faces in a video')
    index_parser.add_argument('video_path', help='Path to video file')
    index_parser.add_argument('--force', action='store_true', help='Force re-indexing even if already indexed')
    
    # Find unmatched command
    unmatched_parser = subparsers.add_parser('find-unmatched', help='Find unmatched faces in indexed video')
    unmatched_parser.add_argument('video_path', help='Path to video file')
    unmatched_parser.add_argument('--reference-faces', required=True, help='Path(s) to reference face images (comma-separated)')
    unmatched_parser.add_argument('--distance-threshold', type=float, default=0.6, 
                                help='Face distance threshold for matching (default: 0.6)')
    unmatched_parser.add_argument('--output', help='Output file for unmatched face report')
    
    # Auto-match command
    automatch_parser = subparsers.add_parser('auto-match', help='Automatically match unmatched faces')
    automatch_parser.add_argument('video_path', help='Path to video file')
    automatch_parser.add_argument('--reference-faces', required=True, help='Path(s) to reference face images (comma-separated)')
    automatch_parser.add_argument('--output', help='Output file for auto-match report')
    
    # Clear cache command
    clear_parser = subparsers.add_parser('clear-cache', help='Clear face cache')
    clear_parser.add_argument('video_path', nargs='?', help='Path to specific video (optional, clears all if not specified)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show cache status')
    status_parser.add_argument('video_path', help='Path to video file')
    
    return parser


def cmd_index_video(args: argparse.Namespace) -> int:
    """Index faces in a video"""
    if not is_video(args.video_path):
        logger.error(f"Invalid video file: {args.video_path}", __name__)
        return 1
    
    logger.info(f"Indexing faces in: {args.video_path}", __name__)
    
    # Check if already indexed
    is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(args.video_path)
    if is_indexed and not args.force:
        logger.info(f"Video already indexed: {metadata['indexed_frames']}/{metadata['total_frames']} frames", __name__)
        return 0
    
    if args.force and is_indexed:
        logger.info("Force re-indexing video...", __name__)
        VIDEO_FACE_INDEX.clear_video_index(args.video_path)
    
    # Progress callback
    def progress_callback(progress: float):
        print(f"\rIndexing progress: {progress:.1%}", end='', flush=True)
    
    try:
        success = VIDEO_FACE_INDEX.index_video_faces(args.video_path, progress_callback)
        print()  # New line after progress
        
        if success:
            _, metadata = VIDEO_FACE_INDEX.is_video_indexed(args.video_path)
            logger.info(f"Indexing complete: {metadata['indexed_frames']}/{metadata['total_frames']} frames", __name__)
            return 0
        else:
            logger.error("Indexing failed", __name__)
            return 1
            
    except Exception as e:
        print()  # New line after progress
        logger.error(f"Error during indexing: {e}", __name__)
        return 1


def cmd_find_unmatched(args: argparse.Namespace) -> int:
    """Find unmatched faces in video"""
    if not is_video(args.video_path):
        logger.error(f"Invalid video file: {args.video_path}", __name__)
        return 1
    
    # Check if video is indexed
    is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(args.video_path)
    if not is_indexed:
        logger.error(f"Video not indexed. Run 'index {args.video_path}' first.", __name__)
        return 1
    
    # Load reference faces
    reference_paths = [path.strip() for path in args.reference_faces.split(',')]
    logger.info(f"Loading reference faces from {len(reference_paths)} images", __name__)
    
    # For CLI, we need to manually set up reference faces
    # This is a simplified version - in practice you'd want to integrate with face_analyser
    reference_faces = {}
    try:
        from facefusion.vision import read_static_images
        from facefusion.face_analyser import get_many_faces, get_one_face
        
        for idx, reference_path in enumerate(reference_paths):
            frames = read_static_images([reference_path])
            if frames:
                face = get_one_face(get_many_faces(frames))
                if face:
                    reference_faces[idx] = face
                    logger.info(f"Loaded reference face {idx} from {reference_path}", __name__)
                else:
                    logger.warn(f"No face found in {reference_path}", __name__)
            else:
                logger.warn(f"Could not load image {reference_path}", __name__)
        
        if not reference_faces:
            logger.error("No valid reference faces loaded", __name__)
            return 1
        
    except Exception as e:
        logger.error(f"Error loading reference faces: {e}", __name__)
        return 1
    
    # Find unmatched faces
    logger.info(f"Finding unmatched faces with threshold {args.distance_threshold}", __name__)
    unmatched_faces = VIDEO_FACE_INDEX.find_unmatched_faces(
        args.video_path, reference_faces, args.distance_threshold
    )
    
    # Report results
    if unmatched_faces:
        logger.info(f"Found unmatched faces in {len(unmatched_faces)} frames", __name__)
        
        # Show summary
        total_unmatched = sum(len(faces) for faces in unmatched_faces.values())
        logger.info(f"Total unmatched faces: {total_unmatched}", __name__)
        
        # Output detailed report if requested
        if args.output:
            try:
                import json
                report = {
                    'video_path': args.video_path,
                    'reference_faces': reference_paths,
                    'distance_threshold': args.distance_threshold,
                    'unmatched_frames': len(unmatched_faces),
                    'total_unmatched_faces': total_unmatched,
                    'frames': {
                        str(frame_num): len(faces) 
                        for frame_num, faces in unmatched_faces.items()
                    }
                }
                
                with open(args.output, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Report saved to {args.output}", __name__)
            except Exception as e:
                logger.error(f"Error saving report: {e}", __name__)
                return 1
    else:
        logger.info("No unmatched faces found", __name__)
    
    return 0


def cmd_auto_match(args: argparse.Namespace) -> int:
    """Automatically match unmatched faces"""
    if not is_video(args.video_path):
        logger.error(f"Invalid video file: {args.video_path}", __name__)
        return 1
    
    # First find unmatched faces (reuse logic from cmd_find_unmatched)
    # This is a simplified version - you'd want to refactor the common code
    logger.info("Finding unmatched faces for auto-matching...", __name__)
    
    # Load reference faces (same as cmd_find_unmatched)
    reference_paths = [path.strip() for path in args.reference_faces.split(',')]
    reference_faces = {}
    
    try:
        from facefusion.vision import read_static_images
        from facefusion.face_analyser import get_many_faces, get_one_face
        
        for idx, reference_path in enumerate(reference_paths):
            frames = read_static_images([reference_path])
            if frames:
                face = get_one_face(get_many_faces(frames))
                if face:
                    reference_faces[idx] = face
        
        if not reference_faces:
            logger.error("No valid reference faces loaded", __name__)
            return 1
            
    except Exception as e:
        logger.error(f"Error loading reference faces: {e}", __name__)
        return 1
    
    # Find unmatched faces
    unmatched_faces = VIDEO_FACE_INDEX.find_unmatched_faces(args.video_path, reference_faces)
    
    if not unmatched_faces:
        logger.info("No unmatched faces found for auto-matching", __name__)
        return 0
    
    # Run auto-matching
    logger.info(f"Running auto-match on {len(unmatched_faces)} frames with unmatched faces", __name__)
    auto_matches = SMART_FACE_MATCHER.auto_match_unmatched_faces(
        args.video_path, unmatched_faces, reference_faces
    )
    
    # Report results
    matched_frames = len([f for f in auto_matches.values() if f])
    total_auto_matches = sum(len(matches) for matches in auto_matches.values())
    
    logger.info(f"Auto-matching complete: {matched_frames} frames with auto-matches", __name__)
    logger.info(f"Total auto-matched faces: {total_auto_matches}", __name__)
    
    # Output report if requested
    if args.output:
        try:
            import json
            report = {
                'video_path': args.video_path,
                'reference_faces': reference_paths,
                'auto_matched_frames': matched_frames,
                'total_auto_matches': total_auto_matches,
                'matches': auto_matches
            }
            
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Auto-match report saved to {args.output}", __name__)
        except Exception as e:
            logger.error(f"Error saving report: {e}", __name__)
            return 1
    
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """Clear face cache"""
    try:
        if args.video_path:
            if not is_video(args.video_path):
                logger.error(f"Invalid video file: {args.video_path}", __name__)
                return 1
            
            VIDEO_FACE_INDEX.clear_video_index(args.video_path)
            logger.info(f"Cache cleared for video: {args.video_path}", __name__)
        else:
            VIDEO_FACE_INDEX.clear_video_index()
            logger.info("All video caches cleared", __name__)
        
        return 0
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", __name__)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show cache status"""
    if not is_video(args.video_path):
        logger.error(f"Invalid video file: {args.video_path}", __name__)
        return 1
    
    try:
        is_indexed, metadata = VIDEO_FACE_INDEX.is_video_indexed(args.video_path)
        
        if is_indexed:
            logger.info(f"Video: {args.video_path}", __name__)
            logger.info(f"Status: Indexed", __name__)
            logger.info(f"Total frames: {metadata['total_frames']}", __name__)
            logger.info(f"Indexed frames: {metadata['indexed_frames']}", __name__)
            logger.info(f"FPS: {metadata['fps']}", __name__)
            logger.info(f"Cache file: {metadata['cache_file']}", __name__)
            
            # Show cache file size if available
            import os
            cache_file_path = VIDEO_FACE_INDEX.cache_dir / metadata['cache_file']
            if cache_file_path.exists():
                size_mb = os.path.getsize(cache_file_path) / (1024 * 1024)
                logger.info(f"Cache size: {size_mb:.1f} MB", __name__)
        else:
            logger.info(f"Video: {args.video_path}", __name__)
            logger.info(f"Status: Not indexed", __name__)
        
        return 0
    except Exception as e:
        logger.error(f"Error checking status: {e}", __name__)
        return 1


def main() -> int:
    """Main CLI entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'index':
        return cmd_index_video(args)
    elif args.command == 'find-unmatched':
        return cmd_find_unmatched(args)
    elif args.command == 'auto-match':
        return cmd_auto_match(args)
    elif args.command == 'clear-cache':
        return cmd_clear_cache(args)
    elif args.command == 'status':
        return cmd_status(args)
    else:
        logger.error(f"Unknown command: {args.command}", __name__)
        return 1


if __name__ == '__main__':
    sys.exit(main()) 