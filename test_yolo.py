import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image

# Make sure the facefusion module can be imported
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(base_dir))  # stable-diffusion-webui dir

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Create simple mock for modules if it doesn't exist
if 'modules' not in sys.modules:
    class MockPaths:
        def __init__(self):
            self.script_path = parent_dir
            self.models_path = os.path.join(parent_dir, 'models')
    
    class MockModules:
        def __init__(self):
            self.paths_internal = MockPaths()
            
    sys.modules['modules'] = MockModules()
    sys.modules['modules.paths_internal'] = MockPaths()

from facefusion import state_manager
state_manager.STATES = {'cli': {}, 'ui': {}}

from facefusion.workers.classes.face_masker import FaceMasker

def test_yolo_model():
    # Initialize state items
    state_manager.init_item('custom_yolo_model', None)
    state_manager.init_item('custom_yolo_confidence', 0.5)
    state_manager.init_item('custom_yolo_radius', 10)
    
    # Create a face masker instance
    face_masker = FaceMasker()
    
    # Print available YOLO models
    from facefusion.uis.components.face_masker import find_yolo_models
    models = find_yolo_models()
    print(f"Found {len(models)} YOLO models:")
    for model in models:
        print(f"  - {model}")
    
    if len(models) == 0:
        print("No models found. Please download some YOLO models to the models/adetailer or models/facefusion/yolo directory.")
        return
    
    # Set a model to test
    model_path = models[0]
    state_manager.set_item('custom_yolo_model', model_path)
    
    # Create a test image
    test_image = np.zeros((512, 512, 3), dtype=np.uint8)
    # Draw a rectangle to test detection
    cv2.rectangle(test_image, (100, 100), (400, 400), (255, 255, 255), -1)
    
    # Convert to PIL image for testing
    pil_image = Image.fromarray(test_image)
    
    # Test prediction
    print(f"Testing YOLO model: {model_path}")
    try:
        result = face_masker.ultralytics_predict(model_path, pil_image)
        
        print(f"Prediction results:")
        print(f"  - Bounding boxes: {len(result.bboxes)}")
        print(f"  - Masks: {len(result.masks)}")
        print(f"  - Confidences: {result.confidences}")
        
        if result.preview:
            # Save preview image
            preview_path = os.path.join(base_dir, "yolo_preview.png")
            result.preview.save(preview_path)
            print(f"Preview image saved to: {preview_path}")
        
        # Test mask creation
        if result.bboxes:
            mask = face_masker.create_custom_mask(test_image)
            if mask is not None:
                # Save mask
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_path = os.path.join(base_dir, "yolo_mask.png")
                mask_image.save(mask_path)
                print(f"Mask image saved to: {mask_path}")
    except Exception as e:
        import traceback
        print(f"Error testing YOLO model: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_model() 