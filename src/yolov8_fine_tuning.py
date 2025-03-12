# yolov8_fine_tuning.py

import cv2
from ultralytics import YOLO
import argparse
import os

# Load YOLOv8 Model

def load_model(model_path=None):
    if model_path:
        print(f"[INFO] Loading model from {model_path}")
        model = YOLO(model_path)
    else:
        print("[INFO] Loading pre-trained YOLOv8 model")
        model = YOLO("yolov8n.pt")  # Use 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'
    return model

# Fine-Tune YOLOv8 on Custom Dataset

def fine_tune(model, data_yaml, epochs):
    print(f"[INFO] Fine-tuning YOLOv8 for {epochs} epochs on dataset: {data_yaml}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project='runs/train',
        name='custom_yolo',
        optimizer='auto'
    )

    print("[INFO] Fine-tuning complete!")
    print(f"[INFO] Best model saved at: {results.save_dir}")


# Detect Objects in Image/Video

def detect_objects(model, input_path, output_path=None):
    print(f"[INFO] Running object detection on: {input_path}")

    # Open webcam or video file
    if input_path == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)

    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)
        annotated_frame = results[0].plot()

        # Display annotated frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Save to output file (if specified)
        if output_path:
            out.write(annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection completed!")


# Export Model to ONNX

def export_model(model, export_path):
    print(f"[INFO] Exporting model to {export_path}")
    model.export(format='onnx', path=export_path)
    print(f"[INFO] Model exported to {export_path}")


# Main Function

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on a custom dataset and run object detection.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'detect', 'export'], help="Mode of operation")
    parser.add_argument('--data', type=str, help="Path to dataset YAML file (required for training)")
    parser.add_argument('--model', type=str, help="Path to pre-trained model or fine-tuned model")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--input', type=str, help="Path to input file for detection ('webcam' for webcam)")
    parser.add_argument('--output', type=str, help="Path to save the output video")
    parser.add_argument('--export_path', type=str, help="Path to export the model")

    args = parser.parse_args()

    model = load_model(args.model)

    if args.mode == 'train':
        if not args.data:
            raise ValueError("[ERROR] Please provide path to dataset YAML for training.")
        fine_tune(model, args.data, args.epochs)

    elif args.mode == 'detect':
        if not args.input:
            raise ValueError("[ERROR] Please provide path to input file or use 'webcam'.")
        detect_objects(model, args.input, args.output)

    elif args.mode == 'export':
        if not args.export_path:
            raise ValueError("[ERROR] Please provide path to export the model.")
        export_model(model, args.export_path)

if __name__ == "__main__":
    main()
