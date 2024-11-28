import torch
from PIL import Image
import cv2
import numpy as np
import os
import random
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Replace with the path to your YOLOv5 model
model_path = r'C:\Users\hafid\OneDrive\Documents\Classification1576\YoloV5\best5.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

def annotate_image(image_path):
    """
    Annotates the given image with bounding boxes and labels using YOLOv5.

    Args:
        image_path (str): Path to the input image to process.

    Returns:
        tuple: Counts of detected "Tree" and "Ponds" objects.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Perform inference
    results = model(image)
    detections = results.pandas().xyxy[0]  # Get detection results
    
    # Count specific objects
    detected_objects = detections['name'].tolist()
    tra = detected_objects.count("Tree")
    pna = detected_objects.count("Ponds")
    
    # Annotate the image
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label, confidence = row['name'], row['confidence']
        
        # Draw bounding box
        cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add label
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(img_cv2, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save annotated image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, img_cv2)
    print(f"Annotated image saved to {output_path}")
    
    return tra, pna
