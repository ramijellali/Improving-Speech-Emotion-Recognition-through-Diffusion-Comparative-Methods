import sys
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import sounddevice as sd
import matplotlib.pyplot as plt
# Use PyQt6 backend for matplotlib
import matplotlib

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QLabel, QWidget, QProgressBar, QComboBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QIcon

# Parameters
TARGET_SAMPLE_RATE = 22025
MAX_AUDIO_LENGTH = 10  # seconds
HOP_LENGTH = 256
N_FFT = 1024
N_MELS = 80
RECORD_SECONDS = 3  # Default recording duration

# Emotion labels
emotion_labels = ['Anger', 'Happiness', 'Sadness', 'Fear', 'Neutrality', 'Disgust']

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ResNet-based Emotion Recognition Model
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_emotions=6):
        super(EmotionRecognitionModel, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Simplified ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Average pooling and classification layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_emotions)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# Load the model
def load_model(model_path):
    try:
        model = EmotionRecognitionModel(num_emotions=len(emotion_labels)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Extract mel-spectrogram from audio
def extract_mel_spectrogram(audio, sr=TARGET_SAMPLE_RATE):
    # Resample if needed
    if sr != TARGET_SAMPLE_RATE:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    # Pad or trim audio to fixed length
    target_length = TARGET_SAMPLE_RATE * MAX_AUDIO_LENGTH
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize using Z-score normalization
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()

    return mel_spec_normalized


# Predict emotion from audio
def predict_emotion(model, audio, sr=TARGET_SAMPLE_RATE):
    try:
        # Extract mel-spectrogram
        mel_spec = extract_mel_spectrogram(audio, sr)

        # Convert to tensor and add batch and channel dimensions
        mel_spec_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(mel_spec_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()

        # Get confidence
        confidence = probabilities[0][predicted_class].item()

        return emotion_labels[predicted_class], confidence, probabilities[0].cpu().numpy()
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        return "Error", 0.0, np.zeros(len(emotion_labels))


# Audio recording thread
class AudioRecorder(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, duration=RECORD_SECONDS, sample_rate=TARGET_SAMPLE_RATE):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate

    def run(self):
        try:
            audio = sd.rec(int(self.duration * self.sample_rate),
                           samplerate=self.sample_rate,
                           channels=1)
            sd.wait()
            self.finished.emit(audio.flatten())
        except Exception as e:
            self.error.emit(str(e))


# Matplotlib canvas for the visualization
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()


# Main application window
class EmotionRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Voice Emotion Recognition")
        self.setGeometry(100, 100, 1000, 600)

        # Load model
        self.model_path = 'emotion_recognition_model.pth'

        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"Error: Model file '{self.model_path}' not found.")
            # Show error in UI later
            self.model = None
        else:
            self.model = load_model(self.model_path)

        # Initialize UI
        self.init_ui()

        # Initialize recorder
        self.recorder = None
        self.recording_duration = RECORD_SECONDS

        # Check model status
        if self.model is None:
            self.status_label.setText(f"Error: Model file '{self.model_path}' not found or could not be loaded.")
            self.record_button.setEnabled(False)

    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Title
        title_label = QLabel("Voice Emotion Recognition")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Top controls
        top_controls = QHBoxLayout()

        # Recording duration
        duration_layout = QHBoxLayout()
        duration_label = QLabel("Recording Duration:")
        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["1 second", "2 seconds", "3 seconds", "5 seconds"])
        self.duration_combo.setCurrentIndex(2)  # Default 3 seconds
        self.duration_combo.currentIndexChanged.connect(self.update_duration)
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_combo)

        # Record button
        self.record_button = QPushButton("Record Voice")
        self.record_button.setFont(QFont("Arial", 12))
        self.record_button.setMinimumHeight(50)
        self.record_button.clicked.connect(self.toggle_recording)

        top_controls.addLayout(duration_layout)
        top_controls.addWidget(self.record_button)

        main_layout.addLayout(top_controls)

        # Recording progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Result display
        self.status_label = QLabel("Press 'Record Voice' to start")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        # Result emotion
        self.emotion_label = QLabel("")
        self.emotion_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.emotion_label)

        # Bar chart
        self.canvas = MplCanvas(self, width=10, height=6)
        main_layout.addWidget(self.canvas)

        # Initialize the bar chart
        self.bars = self.canvas.axes.bar(emotion_labels, [0] * len(emotion_labels), color='skyblue')
        self.canvas.axes.set_ylim(0, 1)
        self.canvas.axes.set_ylabel('Confidence')
        self.canvas.axes.set_title('Emotion Recognition Results')
        self.canvas.axes.tick_params(axis='x', rotation=45)
        self.canvas.fig.tight_layout()

        # Timer for progress bar
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.progress_count = 0

    def update_duration(self, index):
        durations = [1, 2, 3, 5]
        self.recording_duration = durations[index]

    def toggle_recording(self):
        if self.recorder is None or not self.recorder.isRunning():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.record_button.setText("Stop Recording")
        self.status_label.setText(f"Recording for {self.recording_duration} seconds...")
        self.progress_bar.setValue(0)
        self.progress_count = 0

        # Start recording thread
        self.recorder = AudioRecorder(duration=self.recording_duration)
        self.recorder.finished.connect(self.process_audio)
        self.recorder.error.connect(self.handle_recording_error)
        self.recorder.start()

        # Start progress timer
        self.timer.start(int(self.recording_duration * 1000 / 100))  # Update every 1% of duration

    def stop_recording(self):
        self.record_button.setText("Record Voice")
        self.timer.stop()
        self.progress_bar.setValue(100)

        # Stop recording (if possible)
        if self.recorder and self.recorder.isRunning():
            self.recorder.terminate()

    def handle_recording_error(self, error_msg):
        self.record_button.setText("Record Voice")
        self.timer.stop()
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Recording error: {error_msg}")

    def update_progress(self):
        self.progress_count += 1
        self.progress_bar.setValue(self.progress_count)

        if self.progress_count >= 100:
            self.timer.stop()

    def process_audio(self, audio):
        self.record_button.setText("Record Voice")
        self.status_label.setText("Processing audio...")

        # Predict emotion
        emotion, confidence, probabilities = predict_emotion(self.model, audio)

        # Update UI
        self.status_label.setText(f"Detected emotion with {confidence:.1%} confidence")
        self.emotion_label.setText(f"{emotion}")
        self.update_chart(probabilities)

    def update_chart(self, probabilities):
        # Update each bar
        for bar, prob in zip(self.bars, probabilities):
            bar.set_height(prob)

        # Update colors based on the highest probability
        max_index = np.argmax(probabilities)
        colors = ['lightblue'] * len(emotion_labels)
        colors[max_index] = 'red'

        for bar, color in zip(self.bars, colors):
            bar.set_color(color)

        # Redraw the canvas
        self.canvas.draw()


# Run the application
def main():
    app = QApplication(sys.argv)
    window = EmotionRecognitionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()