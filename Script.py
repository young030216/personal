from ultralytics import YOLO
import torch
import os

if __name__ == '__main__':
    print('Starting training...', flush=True)
    
    # Load YOLO model and move it to the correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    
    # Train the model
    model.train(
        data='data/1024_images/1024_consilium.yaml',
        epochs=20,
        imgsz=640, # for now we resize to 640 to see how is the accuracy, and we can further fine-tune it.
        batch=32,
        workers=4,
        device=device
    )
    
    # Validation and testing
    val_results = model.val(
        data='data/1024_images/1024_consilium.yaml'
    )

    print("Training complete. Validation results:")
    print(val_results)
