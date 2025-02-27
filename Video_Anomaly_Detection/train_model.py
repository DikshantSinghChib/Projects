import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Define the dataset directory
dataset_dir = r'C:\Users\DIKSHANT\Desktop\anomaly detection\violence-detection-dataset'

def preprocess_video(video_path, num_frames=30, target_size=(120, 120)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"Error: Video {video_path} has no frames")
        return None
    
    indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()

    if len(frames) < num_frames:
        frames.extend([np.zeros(target_size + (3,), dtype=np.uint8)] * (num_frames - len(frames)))
    
    return np.array(frames)

def process_video_file(args):
    video_path, class_label = args
    video = preprocess_video(video_path)
    if video is not None and video.size > 0:
        return video, class_label
    return None

def load_data(data_dir):
    video_paths = []
    labels = []
    class_labels = {'non-violent': 0, 'violent': 1}

    for class_name, class_label in class_labels.items():
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping...")
            continue
        
        for cam in os.listdir(class_dir):
            cam_dir = os.path.join(class_dir, cam)
            if not os.path.isdir(cam_dir):
                continue
            
            video_files = [f for f in os.listdir(cam_dir) if f.lower().endswith(('.mp4'))]
            for video_file in video_files:
                video_path = os.path.join(cam_dir, video_file)
                video_paths.append((video_path, class_label))

    if not video_paths:
        print("No videos found. Check the directory structure and video formats.")
        return np.array([]), np.array([])
    
    videos = []
    labels = []
    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_video_file, args) for args in video_paths]
        for future in tqdm(as_completed(futures), total=len(video_paths), desc="Processing videos"):
            result = future.result()
            if result is not None:
                video, label = result
                videos.append(video)
                labels.append(label)

    return np.array(videos), np.array(labels)

def create_model(input_shape):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape[1:], include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.TimeDistributed(base_model),
        layers.GlobalAveragePooling3D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    videos, labels = load_data(dataset_dir)
    
    if videos.size == 0 or labels.size == 0:
        print("Error: No data loaded. Exiting.")
        exit()

    print(f"Data loaded. Shape of videos: {videos.shape}, Shape of labels: {labels.shape}")

    # Convert to float32 and normalize
    videos = videos.astype(np.float32) / 255.0

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(videos, labels, test_size=0.2, random_state=42)
    
    print("Creating model...")
    model = create_model(videos.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("Training model...")
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=16, 
                        validation_split=0.2,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    print("Saving model...")
    model.save('violence_detection_model.h5')
    print("Model saved successfully.")