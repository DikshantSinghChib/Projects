import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class ViolenceDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Violence Detection Interface")
        master.geometry("400x200")

        self.model = load_model('violence_detection_model.h5')

        self.label = tk.Label(master, text="Select a video file for violence detection")
        self.label.pack(pady=20)

        self.select_button = tk.Button(master, text="Select Video", command=self.select_video)
        self.select_button.pack(pady=10)

        self.result_label = tk.Label(master, text="")
        self.result_label.pack(pady=20)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            prediction = self.process_video(file_path)
            self.display_result(prediction)

    def process_video(self, video_path):
        frames = self.preprocess_video(video_path)
        if frames is not None:
            prediction = self.model.predict(np.expand_dims(frames, axis=0))
            return prediction[0][0]
        return None

    def preprocess_video(self, video_path, num_frames=30, target_size=(120, 120)):
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            messagebox.showerror("Error", f"Video {video_path} has no frames")
            return None
        
        step = max(frame_count // num_frames, 1)
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
            frame = frame.astype(np.float32) / 255.0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if len(frames) == num_frames:
                break
        
        cap.release()

        if len(frames) < num_frames:
            frames.extend([np.zeros(target_size + (3,), dtype=np.float32)] * (num_frames - len(frames)))
        
        return np.array(frames)

    def display_result(self, prediction):
        if prediction is not None:
            result = "Violence Detected" if prediction > 0.5 else "No Violence Detected"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            self.result_label.config(text=f"{result}\nConfidence: {confidence:.2%}")
        else:
            self.result_label.config(text="Error processing video")

def main():
    root = tk.Tk()
    app = ViolenceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()