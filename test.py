import tkinter as tk
from PIL import Image, ImageTk
import time
import threading
import cv2
import numpy as np
import glob
import random
from ultralytics import YOLO

# Load YOLO model
MODEL_PATH = r"E:\VScode\ASAP Sem 2\runs\detect\train_yolov9\weights\best.pt"
model = YOLO(MODEL_PATH)
model.eval()

# Class names
class_names = ['NEV', 'EV']

# GUI Setup
root = tk.Tk()
root.title("Smart Traffic Management System")
root.geometry("1000x800")
root.configure(bg="lightgray")

# Banner for EV detection
ev_banner = tk.Label(root, text="", font=("Arial", 20, "bold"), fg="white", bg="darkred", height=2)
ev_banner.pack(fill='x')

# Main content frame
main_frame = tk.Frame(root, bg="lightgray")
main_frame.pack(expand=True, fill="both")

# Traffic lanes
lanes = ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']
signal_state = {lane: 'red' for lane in lanes}
vehicle_count = {lane: 0 for lane in lanes}
ev_detected = {lane: False for lane in lanes}
signal_time = {lane: 5 for lane in lanes}

# Lane-specific folders
image_folders = {
    "Lane 1": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane1",
    "Lane 2": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane2",
    "Lane 3": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane3",
    "Lane 4": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane4"
}

# Traffic signal labels and video previews
labels = {}
video_labels = {}

for i, lane in enumerate(lanes):
    row, col = divmod(i, 2)

    labels[lane] = tk.Label(
        main_frame,
        text=f"{lane}: RED | Vehicles: 0 | Time: {signal_time[lane]}s",
        font=("Arial", 14),
        fg='white',
        bg='red',
        padx=15,
        pady=5
    )
    labels[lane].grid(row=row*2, column=col, pady=5, padx=5, sticky="nsew")

    video_labels[lane] = tk.Label(main_frame, bg="black")
    video_labels[lane].grid(row=row*2+1, column=col, pady=5, padx=5, sticky="nsew")

# Configure layout
for i in range(4):
    main_frame.rowconfigure(i, weight=1)
for j in range(2):
    main_frame.columnconfigure(j, weight=1)

def process_images(lane):
    global vehicle_count, ev_detected

    image_files = glob.glob(f"{image_folders[lane]}/*.jpg")
    if not image_files:
        print(f"No images for {lane}")
        return

    image_path = random.choice(image_files)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image for {lane}")
        return

    results = model.predict(frame, imgsz=640, conf=0.3, iou=0.5)
    boxes = results[0].boxes
    vehicle_count[lane] = len(boxes)
    ev_detected[lane] = False

    for box in boxes:
        cls = int(box.cls[0])
        label = class_names[cls]

        if label == "EV":
            ev_detected[lane] = True

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if label == 'EV' else (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if label == 'EV' else (255, 0, 0), 2)

    signal_time[lane] = max(5, min(30, vehicle_count[lane] * 2))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((400, 300))
    imgtk = ImageTk.PhotoImage(image=img)

    video_labels[lane].config(image=imgtk)
    video_labels[lane].image = imgtk
    labels[lane].config(text=f"{lane}: {signal_state[lane].upper()} | Vehicles: {vehicle_count[lane]} | Time: {signal_time[lane]}s")

def update_signals():
    while True:
        # Prioritize EVs
        ev_lanes = [lane for lane in lanes if ev_detected[lane]]
        non_ev_lanes = [lane for lane in lanes if lane not in ev_lanes]
        sorted_lanes = ev_lanes + sorted(non_ev_lanes, key=lambda x: vehicle_count[x], reverse=True)

        if ev_lanes:
            ev_banner.config(text=f"EV Detected in {', '.join(ev_lanes)}! Prioritizing Emergency Vehicle...", bg="darkgreen")
        else:
            ev_banner.config(text="", bg="darkred")

        for lane in sorted_lanes:
            for l in lanes:
                signal_state[l] = 'red'
                labels[l].config(bg='red', text=f"{l}: RED | Vehicles: {vehicle_count[l]} | Time: {signal_time[l]}s")

            signal_state[lane] = 'green'

            for t in range(signal_time[lane], 0, -1):
                labels[lane].config(bg='green', text=f"{lane}: GREEN | Vehicles: {vehicle_count[lane]} | Time: {t}s")
                time.sleep(1)

            process_images(lane)

# Start the thread
threading.Thread(target=update_signals, daemon=True).start()

root.mainloop()
