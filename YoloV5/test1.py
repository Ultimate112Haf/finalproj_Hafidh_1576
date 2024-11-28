import torch
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path
import os
import pathlib
import random
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Replace 'yolov5.pt' with the path to your YOLOv5 model
force_reload=True
model_path = r'C:\Users\hafid\OneDrive\Documents\Classification1576\YoloV5\best5.pt'
print(os.path.abspath(model_path))  # Print the absolute path to verify
print(os.path.exists(model_path))   # Check if the file exists

#model_path = str(Path('best.pt'))
print(os.path.abspath(model_path)) 
#image_path = 'Images/camp2.jpg'
image_path = r'C:\Users\hafid\OneDrive\Documents\Classification1576\YoloV5\Images\camp2.jpg'
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)


def annotate_image(image_path):
    """
    Annotates the given image with bounding boxes and labels using YOLOv5.

    Args:
        image_path (str): Path to the input image to process.
    """
    # Load the image
    image = Image.open(image_path)
    
    # Perform inference
    results = model(image)
    
    # Convert the results to Pandas DataFrame for easy processing
    detections = results.pandas().xyxy[0]  # bounding box coordinates (xmin, ymin, xmax, ymax)
    obj =[]
    obj.extend(results.pandas().xyxy[0].name.tolist())
    print(obj)

    print(obj)  # Print the list to see all object names
    tra = obj.count("Tree")
    pna = obj.count("Ponds")
    print(tra, pna)
    
   
    # Convert the image to a format suitable for OpenCV
    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Annotate the image with bounding boxes and labels
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax, label, confidence = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'], row['confidence']
        
        # Draw bounding box
        cv2.rectangle(img_cv2, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Add label
        label_text = f"{label} ({confidence:.2f})"
        cv2.putText(img_cv2, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    
    # Save or display the annotated image
    # Replace 'output_image.jpg' with the desired output path
    nama=random.randint(1,1000)
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, img_cv2)
    print(f"Annotated image saved to {output_path}")

# Replace with the path to your input image
#image_path = "replace_with_path_to_image.jpg"
annotate_image(image_path)

