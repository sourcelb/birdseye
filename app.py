import os
os.system("pip install opencv-python")
os.system("pip install numpy")
os.system("pip install requests")
os.system("pip install gtts")
os.system("pip install ddg")
os.system("pip install pillow")

# Imports
import cv2
import numpy as np
import requests
from gtts import gTTS
from ddg import ddg
import tkinter as tk
from tkinter import Label, Button, Text
from PIL import Image, ImageTk

# Setup
OBJECT_DETECTION_MODEL = "yolov3.weights"
OBJECT_DETECTION_CONFIG = "yolov3.cfg"
UNALLOWED_FILE = "unallowed.txt"

def load_object_detection_model():
    net = cv2.dnn.readNet(OBJECT_DETECTION_MODEL, OBJECT_DETECTION_CONFIG)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def detect_objects(net, output_layers, img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

def read_unallowed_sites():
    if os.path.exists(UNALLOWED_FILE):
        with open(UNALLOWED_FILE, "r") as file:
            return set(file.read().splitlines())
    return set()

def write_unallowed_site(site):
    with open(UNALLOWED_FILE, "a") as file:
        file.write(site + "\n")

def perform_web_lookup(object_name):
    unallowed_sites = read_unallowed_sites()
    search_results = ddg(object_name, max_results=5)  # Adjust the max_results as needed
    filtered_results = [result for result in search_results if result['href'] not in unallowed_sites]
    return filtered_results

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")  # For Windows; use "afplay output.mp3" on macOS

def capture_image():
    global frame
    _, frame = cap.read()
    cv2.imshow("Captured Image", frame)

def analyze_image():
    global frame
    class_ids, confidences, boxes = detect_objects(net, output_layers, frame)
    if class_ids:
        object_name = "detected_object_name"  # Replace with actual detected object name
        search_results = perform_web_lookup(object_name)
        if search_results:
            snippet = search_results[0].get('body', 'No relevant information found.')
            text_to_speech(snippet)
            result_text.insert(tk.END, snippet + "\n")
            result_text.insert(tk.END, "Source: " + search_results[0]['href'] + "\n")
        else:
            text_to_speech("No relevant information found.")
            result_text.insert(tk.END, "No relevant information found.\n")
    else:
        text_to_speech("No objects detected.")
        result_text.insert(tk.END, "No objects detected.\n")

def flag_incorrect_info():
    incorrect_info = result_text.get("1.0", tk.END).strip().split("\n")[-1]
    write_unallowed_site(incorrect_info)
    result_text.insert(tk.END, "Information flagged as incorrect and will be avoided in future searches.\n")

if __name__ == "__main__":
    # Load object detection model
    net, output_layers = load_object_detection_model()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    # Create GUI
    root = tk.Tk()
    root.title("Object Detection App")

    capture_button = Button(root, text="Capture Image", command=capture_image)
    capture_button.pack()

    analyze_button = Button(root, text="Analyze Image", command=analyze_image)
    analyze_button.pack()

    result_text = Text(root, height=10, width=50)
    result_text.pack()

    flag_button = Button(root, text="Flag Incorrect Info", command=flag_incorrect_info)
    flag_button.pack()

    root.mainloop()

    # Release the camera when done
    cap.release()
    cv2.destroyAllWindows()

