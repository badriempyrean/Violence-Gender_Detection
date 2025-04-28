import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from collections import deque

# ==== AntiFlicker Class ====
class AntiFlicker:
    def __init__(self, mode='majority', window_size=5, num_classes=2):
        self.mode = mode
        self.window_size = window_size
        self.num_classes = num_classes
        self.buffer = deque(maxlen=window_size)
        self.last_prediction = None

    def update(self, prediction):
        self.buffer.append(prediction)
        if self.mode == 'majority':
            votes = list(self.buffer)
            counts = np.bincount(votes, minlength=self.num_classes)
            majority_class = np.argmax(counts)
            self.last_prediction = majority_class
            return self.last_prediction
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def reset(self):
        self.buffer.clear()
        self.last_prediction = None

# ==== Load Models (TFLite) ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load TFLite interpreters
violence_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "violence_detector_float16_flex.tflite"))
gender_interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "gender_classifier_fp16v2.tflite"))

violence_interpreter.allocate_tensors()
gender_interpreter.allocate_tensors()

violence_input = violence_interpreter.get_input_details()
violence_output = violence_interpreter.get_output_details()

gender_input = gender_interpreter.get_input_details()
gender_output = gender_interpreter.get_output_details()

# Load object detector
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

# ==== Constants ====
FRAME_DIM = (128, 128)
SEQUENCE_LENGTH = 16
FRAME_SKIP = 10

# ==== Utility Functions ====
def draw_box(frame, box, label, color=(0, 255, 0)):
    y1, x1, y2, x2 = box
    h, w, _ = frame.shape
    x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def predict_violence(frames):
    frames = [cv2.resize(f, FRAME_DIM) for f in frames]
    frames = np.array(frames) / 255.0
    input_seq = np.expand_dims(frames, axis=0).astype(np.float32)

    violence_interpreter.set_tensor(violence_input[0]['index'], input_seq)
    violence_interpreter.invoke()
    prediction = violence_interpreter.get_tensor(violence_output[0]['index'])
    return prediction[0][0]

def predict_gender(person_array, confidence_threshold=0.8):
    person_array = np.expand_dims(person_array, axis=0).astype(np.float32)
    gender_interpreter.set_tensor(gender_input[0]['index'], person_array)
    gender_interpreter.invoke()
    output = gender_interpreter.get_tensor(gender_output[0]['index'])

    confidence = output[0][0]  # Assuming the model output is a confidence score for female
    gender = "female" if confidence > 0.6 else "male"

    # Only update the prediction if confidence is above threshold
    if confidence < confidence_threshold:
        return None, confidence  # No update to the prediction if confidence is low
    else:
        return gender, confidence

# ==== Video Stream Processor ====
def process_stream(source, panel, root):
    cap = cv2.VideoCapture(source)
    frame_buffer = []
    frame_counter = 0
    violence_prob_display = [0.0]
    gender_filters = {}  # Anti-flicker per person box

    def predict_violence_thread(buffer_copy):
        prob = predict_violence(buffer_copy)
        violence_prob_display[0] = prob

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        female_boxes = []
        male_boxes = []
        resized_frame = cv2.resize(frame, (320, 240))
        img_tensor = tf.convert_to_tensor([resized_frame], dtype=tf.uint8)
        result = detector(img_tensor)
        result = {key: val.numpy() for key, val in result.items()}

        h, w, _ = frame.shape
        for i in range(len(result["detection_scores"][0])):
            score = result["detection_scores"][0][i]
            if score < 0.5:
                continue
            class_id = int(result["detection_classes"][0][i])
            if class_id != 1:
                continue

            y1, x1, y2, x2 = result["detection_boxes"][0][i]
            x1, x2, y1, y2 = int(x1 * w), int(x2 * w), int(y1 * h), int(y2 * h)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            person_resized = cv2.resize(person_crop, (128, 256))
            person_array = person_resized.astype('float32') / 255.0

            # === Anti-Flicker Gender Prediction with Confidence Threshold ===
            raw_gender, confidence = predict_gender(person_array, confidence_threshold=0.8)
            
            box_key = f"{x1}_{y1}_{x2}_{y2}"  # Ensure box_key is assigned here
            if raw_gender is None:  # If confidence is below threshold, retain previous gender
                gender = gender_filters[box_key].last_prediction if box_key in gender_filters else "male"
            else:
                # Anti-flicker mechanism
                if box_key not in gender_filters:
                    gender_filters[box_key] = AntiFlicker(mode='majority', window_size=5, num_classes=2)
                gender_int = 1 if raw_gender == "female" else 0
                stable_gender_int = gender_filters[box_key].update(gender_int)
                gender = "female" if stable_gender_int == 1 else "male"

            box_info = {
                "bbox": (y1 / h, x1 / w, y2 / h, x2 / w),
                "abs_box": (x1, y1, x2, y2)
            }

            if gender == "female":
                female_boxes.append(box_info)
            else:
                male_boxes.append(box_info)

        # Draw female boxes
        for fb in female_boxes:
            x1, y1, x2, y2 = fb["abs_box"]
            draw_box(frame, fb["bbox"], "female", color=(255, 0, 0))

        # Draw male boxes
        for mb in male_boxes:
            draw_box(frame, mb["bbox"], "male", color=(0, 255, 0))

        # Women count
        cv2.putText(frame, f"Women Count: {len(female_boxes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Violence detection every N frames
        if frame_counter % FRAME_SKIP == 0:
            frame_buffer.append(frame)
            if len(frame_buffer) == SEQUENCE_LENGTH:
                buffer_copy = frame_buffer.copy()
                threading.Thread(target=predict_violence_thread, args=(buffer_copy,), daemon=True).start()
                frame_buffer.pop(0)

        # Draw violence probability
        status = "Violence" if violence_prob_display[0] >= 0.5 else "Non-Violent"
        color = (0, 0, 255) if violence_prob_display[0] >= 0.5 else (0, 255, 0)
        cv2.putText(frame, f"Violence Prob: {status} ({violence_prob_display[0]:.2f})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display frame in Tkinter window
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel.config(image=img)
        panel.image = img
        root.update()

        frame_counter += 1

    cap.release()
    gender_filters.clear()

# ==== GUI Setup ====
def start_video_analysis():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv;*.mpeg")])
    if file_path:
        threading.Thread(target=process_stream, args=(file_path, panel, root)).start()

def start_webcam_analysis():
    threading.Thread(target=process_stream, args=(0, panel, root)).start()

root = tk.Tk()
root.title("Violence + Gender Detection App")
root.geometry("1000x700")
root.resizable(False, False)

# Create main frame
main_frame = tk.Frame(root)
main_frame.pack(pady=10)

# Controls frame
controls_frame = tk.Frame(main_frame)
controls_frame.pack(pady=5)

btn_video = tk.Button(controls_frame, text="Analyze Video File", command=start_video_analysis)
btn_video.grid(row=0, column=0, padx=10)

btn_webcam = tk.Button(controls_frame, text="Start Webcam Analysis", command=start_webcam_analysis)
btn_webcam.grid(row=0, column=1, padx=10)

# Preview frame
panel = tk.Label(main_frame)
panel.pack(pady=10)

root.mainloop()
