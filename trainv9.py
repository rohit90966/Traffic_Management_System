from ultralytics import YOLO

def train_yolov9():
    model = YOLO('yolov9c.pt')  # or another YOLOv9 variant
    model.train(
        data='E:/VScode/asep_2_dataset/merged_dataset_last/data.yaml',
        epochs=10,
        imgsz=416,
        batch=8,
        name='train_yolov9_finalmodel2'
    )

# ✅ Required on Windows for multiprocessing to work
if __name__ == '__main__':
    train_yolov9()
