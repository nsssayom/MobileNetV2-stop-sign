import tensorflow as tf
import numpy as np
import cv2
import os
import logging
from datetime import datetime

# Attack Configuration Constants
INITIAL_EPSILON = 0.02          # Starting perturbation strength
MIN_EPSILON = 0.01              # Minimum perturbation strength allowed
MAX_EPSILON = 0.4               # Maximum perturbation strength to keep image visible
EPSILON_MULTIPLIER = 1.5        # How much to increase epsilon when attack is stuck
MAX_ITERATIONS = 100            # Maximum number of attack attempts
TARGET_CONFIDENCE = 0.3         # Attack succeeds if confidence drops below this
PATIENCE = 3                    # Iterations to wait before increasing strength
AGGRESSIVE_THRESHOLD = 20       # Iterations before switching to aggressive mode
SIGNIFICANT_IMPROVEMENT = 0.01   # Minimum improvement to be considered progress
EDGE_PRESERVATION = 0.7         # How much to preserve edges in normal mode
AGGRESSIVE_EDGE_PRESERVATION = 0.3  # Edge preservation in aggressive mode
COLOR_PRESERVATION = 0.5        # How much to preserve red colors normally
AGGRESSIVE_COLOR_BOOST = 1.5    # Extra noise for red regions in aggressive mode
STOP_SIGN_CLASS_ID = 13        # COCO dataset class ID for stop sign
MODEL_PATH = '.tmp/models/ssdlite_mobilenet_v2_coco_2018_05_09/saved_model'  # Path to detection model
INPUT_SIZE = (320, 320)        # Required input size for the model
RED_HSV_RANGES = [             # HSV color ranges for detecting red
    (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Lower red range
    (np.array([170, 100, 100]), np.array([180, 255, 255]))    # Upper red range
]

logging.basicConfig(level=logging.INFO)

class StopSignAttacker:
    def __init__(self):
        self.model = self.load_model()
        
    def load_model(self):
        model = tf.saved_model.load(MODEL_PATH)
        return model.signatures['serving_default']
    
    def get_stop_sign_confidence(self, image):
        input_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)[tf.newaxis, ...]
        pred = self.model(input_tensor)
        
        classes = pred['detection_classes'][0].numpy()
        scores = pred['detection_scores'][0].numpy()
        stop_sign_scores = scores[classes == STOP_SIGN_CLASS_ID]
        
        max_score_idx = np.argmax(scores)
        max_class = int(classes[max_score_idx])
        max_score = scores[max_score_idx]
        
        stop_confidence = np.max(stop_sign_scores) if len(stop_sign_scores) > 0 else 0
        return stop_confidence, max_class, max_score

    def apply_targeted_noise(self, image, current_epsilon, aggressive=False):
        noise = np.random.normal(0, 255 * current_epsilon, image.shape).astype(np.float32)  # Base noise
        
        edges = cv2.Canny(image.astype(np.uint8), 100, 200)  # Detect edges
        edge_mask = edges.astype(np.float32) / 255.0
        edge_mask = np.stack([edge_mask] * 3, axis=-1)
        
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)  # Detect red regions
        red_masks = [cv2.inRange(hsv, low, high) for low, high in RED_HSV_RANGES]
        red_mask = sum(red_masks).astype(np.float32) / 255.0
        red_mask = np.stack([red_mask] * 3, axis=-1)
        
        if aggressive:  # Apply aggressive noise in stuck cases
            preservation_mask = edge_mask * AGGRESSIVE_EDGE_PRESERVATION
            noise = noise * (1.0 + red_mask * AGGRESSIVE_COLOR_BOOST)
        else:  # Normal noise application
            preservation_mask = np.maximum(edge_mask, red_mask * COLOR_PRESERVATION)
            noise = noise * (1 - preservation_mask * EDGE_PRESERVATION)
        
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def process_image(self, image_path):
        print(f"\n{'='*50}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*50}")
        
        img = cv2.imread(image_path)  # Load and prepare image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        
        orig_stop_conf, orig_max_class, orig_max_score = self.get_stop_sign_confidence(img)
        print(f"Original stop sign confidence: {orig_stop_conf:.4f}")
        print(f"Original max class: {orig_max_class} ({orig_max_score:.4f})")
        
        best_image = img.copy()
        best_stop_conf = orig_stop_conf
        current_epsilon = INITIAL_EPSILON
        no_improvement = 0
        aggressive_mode = False
        last_significant_improvement = 0
        
        for i in range(MAX_ITERATIONS):
            if i - last_significant_improvement > AGGRESSIVE_THRESHOLD:  # Switch to aggressive if stuck
                aggressive_mode = True
                current_epsilon = min(current_epsilon * EPSILON_MULTIPLIER, MAX_EPSILON)
                last_significant_improvement = i
                print(f"Switching to aggressive mode. Epsilon: {current_epsilon:.4f}")
            
            noisy_image = self.apply_targeted_noise(best_image, current_epsilon, aggressive_mode)
            stop_conf, max_class, max_score = self.get_stop_sign_confidence(noisy_image)
            
            improvement = best_stop_conf - stop_conf
            if improvement > SIGNIFICANT_IMPROVEMENT:  # Update if significantly better
                print(f"Iteration {i+1}: Improved stop sign confidence: {stop_conf:.4f} "
                      f"(max class: {max_class}, conf: {max_score:.4f})")
                best_stop_conf = stop_conf
                best_image = noisy_image.copy()
                no_improvement = 0
                last_significant_improvement = i
                
                if not aggressive_mode:
                    current_epsilon = max(MIN_EPSILON, current_epsilon * 0.9)
            else:
                no_improvement += 1
            
            if no_improvement >= PATIENCE:  # Increase strength if stuck
                if current_epsilon < MAX_EPSILON:
                    current_epsilon = min(current_epsilon * EPSILON_MULTIPLIER, MAX_EPSILON)
                    print(f"Increasing epsilon to {current_epsilon:.4f}")
                no_improvement = 0
            
            if best_stop_conf < TARGET_CONFIDENCE or (max_class != STOP_SIGN_CLASS_ID and max_score > 0.3):
                print(f"Successfully reduced stop sign confidence at iteration {i+1}")
                break
        
        final_stop_conf, final_max_class, final_max_score = self.get_stop_sign_confidence(best_image)
        
        output_path = os.path.join('noisy_images', f"noisy_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, cv2.cvtColor(best_image, cv2.COLOR_RGB2BGR))
        
        print(f"\nFinal Results:")
        print(f"Original stop sign confidence: {orig_stop_conf:.4f}")
        print(f"Final stop sign confidence: {final_stop_conf:.4f}")
        print(f"New max class: {final_max_class} ({final_max_score:.4f})")
        print(f"Final epsilon: {current_epsilon:.4f}")
        print(f"Saved to: {output_path}")
        
        return {
            'filename': os.path.basename(image_path),
            'original_stop_confidence': orig_stop_conf,
            'final_stop_confidence': final_stop_conf,
            'original_max_class': orig_max_class,
            'final_max_class': final_max_class,
            'final_max_confidence': final_max_score,
            'epsilon_used': current_epsilon,
            'aggressive_mode_used': aggressive_mode
        }

def process_directory(input_dir='input_images', output_dir='noisy_images'):
    os.makedirs(output_dir, exist_ok=True)
    attacker = StopSignAttacker()
    
    results = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            try:
                result = attacker.process_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    if results:
        report_path = os.path.join(output_dir, f'attack_report_{datetime.now():%Y%m%d_%H%M%S}.log')
        with open(report_path, 'w') as f:
            f.write("Attack Results Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"File: {result['filename']}\n")
                f.write(f"Original stop confidence: {result['original_stop_confidence']:.4f}\n")
                f.write(f"Final stop confidence: {result['final_stop_confidence']:.4f}\n")
                f.write(f"Original max class: {result['original_max_class']}\n")
                f.write(f"Final max class: {result['final_max_class']} "
                       f"({result['final_max_confidence']:.4f})\n")
                f.write(f"Epsilon used: {result['epsilon_used']:.4f}\n")
                f.write(f"Used aggressive mode: {result['aggressive_mode_used']}\n")
                f.write("-" * 30 + "\n\n")
            
            avg_drop = sum(r['original_stop_confidence'] - r['final_stop_confidence'] 
                          for r in results) / len(results)
            f.write(f"\nOverall Statistics:\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Average confidence reduction: {avg_drop:.4f}\n")
        
        print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    process_directory()