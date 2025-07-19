
import tkinter as tk
from PIL import Image, ImageTk
import time
import threading
import cv2
import numpy as np
import glob
import random
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import os

# Load YOLO model
MODEL_PATH = r"E:\VScode\ASAP Sem 2\runs\detect\train_yolov9_finalmodel\weights\best.pt"
model = YOLO(MODEL_PATH)
model.eval()

class_names = ['NEV', 'EV']

root = tk.Tk()
root.title("Smart Traffic Management System")
root.geometry("1000x800")
root.configure(bg="lightgray")

ev_banner = tk.Label(root, text="", font=("Arial", 20, "bold"), fg="white", bg="darkred", height=2)
ev_banner.pack(fill='x')

main_frame = tk.Frame(root, bg="lightgray")
main_frame.pack(expand=True, fill="both")

lanes = ['Lane 1', 'Lane 2', 'Lane 3', 'Lane 4']
signal_state = {lane: 'red' for lane in lanes}
vehicle_count = {lane: 0 for lane in lanes}
ev_detected = {lane: False for lane in lanes}
signal_time = {lane: 5 for lane in lanes}
lane_wait_time = {lane: 0 for lane in lanes}
lane_skip_count = {lane: 0 for lane in lanes}  # for decay memory

image_folders = {
    "Lane 1": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane2",
    "Lane 2": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane3",
    "Lane 3": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane4",
    "Lane 4": r"E:\VScode\ASAP Sem 2\dataset2\Vehicle_Detection_Image_Dataset\Test\lane5"
}

labels = {}
video_labels = {}
for i, lane in enumerate(lanes):
    row, col = divmod(i, 2)
    labels[lane] = tk.Label(main_frame, text=f"{lane}: RED | Vehicles: 0 | Time: {signal_time[lane]}s", font=("Arial", 14), fg='white', bg='red', padx=15, pady=5)
    labels[lane].grid(row=row*2, column=col, pady=5, padx=5, sticky="nsew")
    video_labels[lane] = tk.Label(main_frame, bg="black")
    video_labels[lane].grid(row=row*2+1, column=col, pady=5, padx=5, sticky="nsew")

for i in range(4):
    main_frame.rowconfigure(i, weight=1)
for j in range(2):
    main_frame.columnconfigure(j, weight=1)

def log_data_to_excel(lane, count, ev_status):
    log_file = "traffic_log.xlsx"
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "Timestamp": [timestamp],
        "Lane": [lane],
        "Vehicle Count": [count],
        "Emergency Vehicle Detected": ["Yes" if ev_status else "No"]
    }
    df = pd.DataFrame(data)
    if not os.path.exists(log_file):
        df.to_excel(log_file, index=False, engine='openpyxl')
    else:
        with pd.ExcelWriter(log_file, mode='a', if_sheet_exists='overlay', engine='openpyxl') as writer:
            reader = pd.read_excel(log_file)
            df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)

def process_images(lane):
    global vehicle_count, ev_detected
    image_files = glob.glob(f"{image_folders[lane]}/*.jpg")
    if not image_files:
        return
    image_path = random.choice(image_files)
    frame = cv2.imread(image_path)
    if frame is None:
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
        color = (0, 255, 0) if label == 'EV' else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    signal_time[lane] = max(5, min(30, vehicle_count[lane] * 2))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize((400, 300))
    imgtk = ImageTk.PhotoImage(image=img)
    video_labels[lane].config(image=imgtk)
    video_labels[lane].image = imgtk
    labels[lane].config(text=f"{lane}: {signal_state[lane].upper()} | Vehicles: {vehicle_count[lane]} | Time: {signal_time[lane]}s")
    log_data_to_excel(lane, vehicle_count[lane], ev_detected[lane])

def update_signals():
    W1, W2, W3 = 0.6, 0.25, 0.15
    while True:
        # Update all lane images first
        for lane in lanes:
            process_images(lane)

        # Emergency override logic
        active_ev_lanes = [lane for lane in lanes if ev_detected[lane]]
        if active_ev_lanes:
            ev_banner.config(text=f"EV Detected in {', '.join(active_ev_lanes)}!", bg="darkgreen")
            for lane in lanes:
                signal_state[lane] = 'yellow'
                labels[lane].config(bg='orange', text=f"{lane}: YELLOW")
            time.sleep(5)

            best_lane = max(active_ev_lanes, key=lambda l: W1*1 + W2*(lane_wait_time[l]/60) + W3*(vehicle_count[l]/30))
        else:
            ev_banner.config(text="", bg="darkred")
            def lane_score(l):
                ev_score = 1 if ev_detected[l] else 0
                wait_score = min(lane_wait_time[l] / 60, 1)
                veh_score = min(vehicle_count[l] / 30, 1)
                bonus = 0.1 if lane_skip_count[l] >= 3 else 0
                if lane_wait_time[l] >= 90:
                    return 1.5  # starvation override
                return W1*ev_score + W2*wait_score + W3*veh_score + bonus
            best_lane = max(lanes, key=lane_score)

        for lane in lanes:
            signal_state[lane] = 'red'
            labels[lane].config(bg='red', text=f"{lane}: RED | Vehicles: {vehicle_count[lane]} | Time: {signal_time[lane]}s")

        signal_state[best_lane] = 'green'
        lane_wait_time[best_lane] = 0
        lane_skip_count[best_lane] = 0

        for lane in lanes:
            if lane != best_lane:
                lane_wait_time[lane] += signal_time[best_lane]
                lane_skip_count[lane] += 1

        for t in range(signal_time[best_lane], 0, -1):
            labels[best_lane].config(bg='green', text=f"{best_lane}: GREEN | Vehicles: {vehicle_count[best_lane]} | Time: {t}s")
            time.sleep(1)

threading.Thread(target=update_signals, daemon=True).start()
root.mainloop()
