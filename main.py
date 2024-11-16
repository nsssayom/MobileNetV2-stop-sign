import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pathlib
import math
import requests
import tarfile
import os
from typing import List, Tuple
import shutil
import logging
import hashlib
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('detection.log')
    ]
)

# Configure TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Enable GPU memory growth

# Directory structure setup
BASE_DIR = pathlib.Path(os.getcwd())

INPUT_DIR = BASE_DIR / "input_images"
# INPUT_DIR = BASE_DIR / "noisy_images"

OUTPUT_DIR = BASE_DIR / "output_images"

MODEL_DIR = BASE_DIR / "model"
MODEL_CACHE = MODEL_DIR / "model_cache"
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png'}

# Model information
MODEL_INFO = {
    'url': 'http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz',
    'filename': 'ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz',
    'extracted_dir': 'ssdlite_mobilenet_v2_coco_2018_05_09',
    'model_dir': 'saved_model'
}

class GPUCheck:
    """Check and configure GPU settings"""
    @staticmethod
    def setup_gpu():
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"Found {len(gpus)} GPU(s). GPU support enabled.")
                return True
            else:
                logging.warning("No GPU found. Running on CPU.")
                return False
        except Exception as e:
            logging.error(f"Error setting up GPU: {str(e)}")
            return False

class DirectoryManager:
    """Manage directory creation and validation"""
    @staticmethod
    def setup_directories():
        """Create necessary directories if they don't exist"""
        try:
            for dir_path in [INPUT_DIR, OUTPUT_DIR, MODEL_DIR, MODEL_CACHE]:
                dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Directories created/verified at {BASE_DIR}")
            return True
        except Exception as e:
            logging.error(f"Error creating directories: {str(e)}")
            return False

    @staticmethod
    def validate_input_directory():
        """Check if input directory has valid images"""
        try:
            image_files = [f for f in INPUT_DIR.iterdir() 
                          if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
            if not image_files:
                logging.error(f"No supported images found in {INPUT_DIR}")
                logging.info(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
                return False
            logging.info(f"Found {len(image_files)} valid image(s): {[f.name for f in image_files]}")
            return True
        except Exception as e:
            logging.error(f"Error validating input directory: {str(e)}")
            return False

class ModelManager:
    """Handle model downloading and loading"""
    @staticmethod
    def download_model():
        """Download and extract the model if needed"""
        model_file = MODEL_CACHE / MODEL_INFO['filename']
        extract_dir = MODEL_DIR / MODEL_INFO['extracted_dir']
        saved_model_path = extract_dir / MODEL_INFO['model_dir']

        try:
            # Check if model is already properly extracted
            if saved_model_path.exists() and saved_model_path.is_dir():
                if list(saved_model_path.glob('*')):  # Check if directory is not empty
                    logging.info("Model already exists and appears valid, skipping download")
                    return saved_model_path
                else:
                    logging.warning("Found empty model directory, re-downloading")

            # Download the model
            logging.info("Downloading model...")
            response = requests.get(MODEL_INFO['url'], stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            total_size = int(response.headers.get('content-length', 0))

            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            
            with open(model_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            # Extract the model
            logging.info("Extracting model...")
            with tarfile.open(model_file, 'r:gz') as tar:
                tar.extractall(path=MODEL_DIR)

            # Verify extraction
            if not saved_model_path.exists() or not list(saved_model_path.glob('*')):
                raise ValueError("Model extraction failed or directory is empty")

            logging.info("Model successfully downloaded and extracted")
            return saved_model_path

        except Exception as e:
            logging.error(f"Error in model download/extraction: {str(e)}")
            raise

    @staticmethod
    def load_model():
        """Load the detection model with GPU support"""
        try:
            saved_model_path = ModelManager.download_model()
            logging.info(f"Loading model from {saved_model_path}...")
            
            # Load model with GPU support if available
            with tf.device('/GPU:0' if GPUCheck.setup_gpu() else '/CPU:0'):
                model = tf.saved_model.load(str(saved_model_path))
                return model.signatures['serving_default']
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

class ImageProcessor:
    """Handle image processing operations"""
    @staticmethod
    def create_image_grid(images: List[np.ndarray], max_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
        """Create a grid of images with proper sizing"""
        if not images:
            raise ValueError("No images provided for grid creation")

        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            aspect = w / h
            if aspect > 1:
                new_w = min(max_size[0], w)
                new_h = int(new_w / aspect)
            else:
                new_h = min(max_size[1], h)
                new_w = int(new_h * aspect)
            resized = cv2.resize(img, (new_w, new_h))
            resized_images.append(resized)
        
        cell_height = max(img.shape[0] for img in resized_images)
        cell_width = max(img.shape[1] for img in resized_images)
        grid = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)
        
        for idx, img in enumerate(resized_images):
            i = idx // cols
            j = idx % cols
            h, w = img.shape[:2]
            y_offset = i * cell_height + (cell_height - h) // 2
            x_offset = j * cell_width + (cell_width - w) // 2
            grid[y_offset:y_offset + h, x_offset:x_offset + w] = img
        
        return grid

    @staticmethod
    def detect_objects(image: np.ndarray, model) -> dict:
        """Perform object detection on an image"""
        try:
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]

            output_dict = model(input_tensor)
            num_detections = int(output_dict.pop('num_detections'))
            
            output_dict = {key: value[0, :num_detections].numpy() 
                          for key, value in output_dict.items()}
            
            output_dict['num_detections'] = num_detections
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

            return output_dict

        except Exception as e:
            logging.error(f"Error during object detection: {str(e)}")
            raise

    @staticmethod
    def calculate_font_scale(image_width: int) -> float:
        """Calculate appropriate font scale based on image width"""
        base_width = 800  # Reference width
        min_scale = 0.5   # Minimum font scale
        max_scale = 2.0   # Maximum font scale
        
        # Calculate scale relative to base width
        scale = (image_width / base_width) * 1.0
        
        # Clamp scale between min and max values
        return max(min_scale, min(scale, max_scale))


    @staticmethod
    def draw_detections(image: np.ndarray, detections: dict, labels: dict) -> np.ndarray:
        """Draw detection boxes and labels on the image with improved visualization"""
        image_with_detections = image.copy()
        height, width, _ = image_with_detections.shape

        # Calculate base font scale based on image width
        base_width = 800
        font_scale = max(0.5, min(2.0, (width / base_width) * 1.0))
        thickness = max(1, int(font_scale * 2))

        # Colors for different classes (BGR format)
        colors = {
            'stop sign': (0, 0, 255),    # Red for stop signs
            'person': (0, 255, 0),       # Green for people
            'car': (255, 0, 0),          # Blue for cars
            'default': (255, 255, 0)     # Yellow for others
        }

        # Process detections
        for i in range(len(detections['detection_scores'])):
            score = detections['detection_scores'][i]
            class_id = int(detections['detection_classes'][i])
            label = labels.get(class_id, f'Class {class_id}')
            
            # Adjust confidence threshold based on class
            confidence_threshold = 0.35 if label.lower() == 'stop sign' else 0.5
            
            if score > confidence_threshold:
                box = detections['detection_boxes'][i]
                confidence = int(score * 100)
                label_text = f'{label} {confidence}%'

                # Convert normalized coordinates to pixel coordinates
                y1, x1, y2, x2 = [int(coord * (height if i % 2 == 0 else width)) 
                                for i, coord in enumerate(box)]

                # Choose color based on label
                color = colors.get(label.lower(), colors['default'])

                # Draw the bounding box
                cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), color, thickness)

                # Calculate text size for the label
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    thickness
                )

                # Calculate label background dimensions
                padding = int(font_scale * 5)  # Padding around text
                label_bg_width = text_width + (padding * 2)
                label_bg_height = text_height + (padding * 2)

                # Calculate label background position
                bg_x1 = x1
                bg_y1 = max(0, y1 - label_bg_height)  # Ensure it doesn't go above image
                bg_x2 = bg_x1 + label_bg_width
                bg_y2 = y1

                # Draw semi-transparent background for label
                overlay = image_with_detections.copy()
                cv2.rectangle(
                    overlay,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    color,
                    -1
                )
                cv2.addWeighted(overlay, 0.7, image_with_detections, 0.3, 0, image_with_detections)

                # Draw the label text
                text_x = bg_x1 + padding
                text_y = bg_y2 - padding
                cv2.putText(
                    image_with_detections,
                    label_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),  # White text
                    thickness,
                    cv2.LINE_AA
                )

                # Draw confidence bar
                bar_height = max(3, int(font_scale * 3))
                bar_width = label_bg_width
                bar_x = bg_x1
                bar_y = bg_y1 - bar_height - 2

                if bar_y >= 0:  # Only draw if it fits in the image
                    # Background (gray) bar
                    cv2.rectangle(
                        image_with_detections,
                        (bar_x, bar_y),
                        (bar_x + bar_width, bar_y + bar_height),
                        (128, 128, 128),
                        -1
                    )

                    # Filled portion of bar
                    filled_width = int(bar_width * score)
                    cv2.rectangle(
                        image_with_detections,
                        (bar_x, bar_y),
                        (bar_x + filled_width, bar_y + bar_height),
                        color,
                        -1
                    )

        return image_with_detections
    
def main():
    """Main execution function"""
    try:
        # Print system information
        logging.info(f"Python version: {sys.version}")
        logging.info(f"TensorFlow version: {tf.__version__}")
        logging.info(f"OpenCV version: {cv2.__version__}")
        logging.info(f"Current working directory: {os.getcwd()}")
        
        # Check GPU availability
        GPUCheck.setup_gpu()
        
        # Initialize directories
        if not DirectoryManager.setup_directories():
            return

        # Validate input directory
        if not DirectoryManager.validate_input_directory():
            return

        # Load model and labels
        model = ModelManager.load_model()
        labels = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
            49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
            54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
            59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
            64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
            72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
            77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
            82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
            88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
        }

        # Process images
        input_images = []
        output_images = []
        image_paths = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in SUPPORTED_FORMATS]

        for path in tqdm(image_paths, desc="Processing images"):
            try:
                logging.info(f"Processing image: {path.name}")
                
                # Read image
                image = Image.open(path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_np = np.array(image)
                
                input_images.append(image_np)
                
                # Detect and draw
                detections = ImageProcessor.detect_objects(image_np, model)
                image_with_detections = ImageProcessor.draw_detections(image_np, detections, labels)
                output_images.append(image_with_detections)
                
                # Save individual result
                output_path = OUTPUT_DIR / f"{path.stem}_output.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(image_with_detections, cv2.COLOR_RGB2BGR))
                logging.info(f"Saved detection result: {output_path.name}")
                
            except Exception as e:
                logging.error(f"Error processing {path.name}: {str(e)}")
                continue

        # Create and save grids
        if input_images and output_images:
            input_grid = ImageProcessor.create_image_grid(input_images)
            output_grid = ImageProcessor.create_image_grid(output_images)
            
            cv2.imwrite(str(OUTPUT_DIR / "input_grid.png"), cv2.cvtColor(input_grid, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUTPUT_DIR / "output_grid.png"), cv2.cvtColor(output_grid, cv2.COLOR_RGB2BGR))
            logging.info("Processing completed successfully!")
        else:
            logging.error("No images were processed successfully")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()