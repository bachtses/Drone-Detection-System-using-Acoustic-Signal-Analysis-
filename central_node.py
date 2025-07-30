import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import pyaudio
import time
import threading
import matplotlib.pyplot as plt
import io
from matplotlib.patches import Rectangle  
import matplotlib
matplotlib.use('Agg')   
from flask import Flask, Response, render_template_string
import usb.core
import usb.util
import usb.backend.libusb1
from tuning import Tuning
import socket
import json
import requests
from datetime import datetime

####################################################################################
####################                  CONSTANTS                 ####################
####################################################################################
AGGREGATOR_LISTEN_PORT = XXXX

ENABLE_VISUALIZATION = True  

SLIDING_SPEED = 0.3
CONF_HISTORY_LEN = 50
SAMPLE_RATE = 24000
AUDIO_LENGTH = 2
N_MFCC = 40
CHUNK = 1024
STEP_SIZE = int(SLIDING_SPEED * SAMPLE_RATE)  
audio_buffer = np.zeros(SAMPLE_RATE * AUDIO_LENGTH, dtype=np.float32)
buffer_lock = threading.Lock()
central_node_prediction_label = "Background"
central_node_probability = 0
confidence_history = []
latest_angle = -1
angle_lock = threading.Lock()


aggregator_data = {}
aggregator_timestamp = 0
aggregator_lock = threading.Lock()


####################################################################################
####################            SEND JSON TO PLATFORM            ###################
####################################################################################
JSON_ENDPOINT = "https://server.XXXXX"
JSON_PASSWORD = "XXXXXXXX"

def send_json(node, probability, direction, angle, timestamp):
    payload = {
        "password": JSON_PASSWORD,
        "station_id": 1,
        "sensor_id": 7,
        "type": "acoustic_fusion",
        "data": [
            {
                "acoustic": {
                    "station_id": 1,
                    "sensor_id": 2,
                    "detected": 1,
                    "node": node,
                    "confidence": float(f"{probability:.2f}"),
                    "angle": int(angle),
                    "direction": direction,
                    "timestamp": timestamp,
                },
                "objectId": 1,
                "detected": 1,
                "confidence": float(probability),
                "unknown": True
            }
        ]
    }

    # 🔍 Print the JSON before sending
    print("\033[93m[DEBUG] JSON Payload Preview:\033[0m")
    print(json.dumps(payload, indent=2))

    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(JSON_ENDPOINT, json=payload, headers=headers, timeout=10)
        print(f"\033[94m[JSON] Drone detection sent | Node: {node} | Confidence: {probability:.2f} | Angle: {angle}° | Time: {timestamp} | Status: {response.status_code}\033[0m")
    except Exception as e:
        print(f"\033[91m[JSON] Error sending detection: {e}\033[0m")




####################################################################################
####################                LOAD MODEL                  ####################
####################################################################################
try:
    model = tf.keras.models.load_model('model.h5')
    print("\033[96m[Model] Model loaded successfully.\033[0m")
except Exception as e:
    print("\033[91m[Model] Failed to load the model:\033[0m", e)
    exit(1)

####################################################################################
####################          INITIALIZE AUDIO STREAM           ####################
####################################################################################
p = pyaudio.PyAudio()
device_index = None
respeaker_keyword = "respeaker 4 mic array" 
print("\033[96m[Microphone] Checking for ReSpeaker microphone device:\033[0m")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if dev_info['maxInputChannels'] > 0 and respeaker_keyword in dev_info['name'].lower():
        device_index = i
        print(f"\033[96m[Microphone] Found ReSpeaker device: index {i}, name: {dev_info['name']}\033[0m")
        break
if device_index is None:
    print("\033[91m[Microphone] Error: ReSpeaker microphone not found. Please check that it is connected.\033[0m")
    p.terminate()
    exit(1)
else:
    print(f"\033[96m[Microphone] Using device index: {device_index}\033[0m")
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK)
print("\033[96m[Microphone] Audio stream opened. Listening for real-time audio from ReSpeaker microphone...\033[0m")


####################################################################################
####################             FEATURE EXTRACTION             ####################
####################################################################################
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    return (mfcc_mean - np.mean(mfcc_mean)) / np.std(mfcc_mean)


####################################################################################
####################               PREDICTION LOOP              ####################
####################################################################################
def prediction_loop():
    global audio_buffer, central_node_prediction_label, central_node_probability

    while True:
        new_frames = []
        num_new_chunks = int(STEP_SIZE / CHUNK)
        for _ in range(num_new_chunks):
            data = stream.read(CHUNK, exception_on_overflow=False)
            new_frames.append(np.frombuffer(data, dtype=np.float32))
        new_audio = np.concatenate(new_frames)

        with buffer_lock:
            audio_buffer = np.roll(audio_buffer, -len(new_audio))
            audio_buffer[-len(new_audio):] = new_audio

        # Make prediction
        features = extract_mfcc_features(audio_buffer, SAMPLE_RATE)
        features = features.reshape(1, features.shape[0], 1)
        prediction = model.predict(features, verbose=0)
        central_node_prediction_label = "Drone" if prediction[0][0] > 0.7 else "Background"
        central_node_probability = prediction[0][0]
        central_node_timestamp = datetime.now().strftime('%H:%M:%S')

        # Check if aggregator is active
        with aggregator_lock:
            aggregator_active = (time.time() - aggregator_timestamp < 2)
            aggregator_prob = aggregator_data.get("probability", 0)
            aggregator_angle = aggregator_data.get("angle", -1)
            aggregator_node = aggregator_data.get("node", "Unknown")
            aggregator_direction = aggregator_data.get("direction", "?")
            aggregator_time = aggregator_data.get("timestamp", "?")

        # Update confidence history
        if not aggregator_active or central_node_prediction_label == "Drone":
            with buffer_lock:
                confidence_history.append(central_node_probability)
                if len(confidence_history) > CONF_HISTORY_LEN:
                    confidence_history.pop(0)

        with angle_lock:
            central_angle = latest_angle

        # Print central node status
        if central_node_prediction_label == 'Background':
            print(f"[Central Node] No Drone")
        else:
            print(f"[Central Node] Drone | Node C | {central_node_probability:.2f} | C | {central_angle}° | {central_node_timestamp}")

        # Print aggregator status
        if aggregator_active and aggregator_prob > 0.7:
            print(f"[Aggregator]   Drone | {aggregator_node} | {aggregator_prob:.2f} | {aggregator_direction} | {aggregator_angle}° | {aggregator_time}")
        else:
            print("[Aggregator]   No Drone")

        # === FINAL FUSION ===
        central_detected = central_node_prediction_label == "Drone" and central_node_probability > 0.7
        aggregator_detected = aggregator_active and aggregator_prob > 0.7

        if aggregator_detected or central_detected:
            if aggregator_detected and (not central_detected or aggregator_prob >= central_node_probability):
                fusion_node = aggregator_node 
                fusion_probability = aggregator_prob
                fusion_direction = aggregator_direction
                fusion_angle = aggregator_angle
                fusion_timestamp = aggregator_time
            else:
                fusion_node = "Central Node"
                fusion_probability = central_node_probability
                fusion_direction = "C"
                fusion_angle = central_angle
                fusion_timestamp = central_node_timestamp

            print(f"\033[92m[FUSION] Drone Detected | Node: {fusion_node} | Confidence: {fusion_probability:.2f} | Direction: {fusion_direction} | Angle: {fusion_angle}° | Timestamp: {fusion_timestamp}\033[0m")

            threading.Thread(
                target=send_json,
                args=(fusion_node, fusion_probability, fusion_direction, fusion_angle, fusion_timestamp),
                daemon=True
            ).start()       

        print("\n═════════════════════════════════════════════════════════════════════\n")


####################################################################################
####################              AGGREGATOR LISTENER           ####################
####################################################################################
def aggregator_listener():
    global aggregator_data, aggregator_timestamp

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', AGGREGATOR_LISTEN_PORT))
    server_socket.listen(1)

    print(f"\033[96m[Aggregator] Ready for incoming aggregator connection on port {AGGREGATOR_LISTEN_PORT}...\033[0m")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"\033[96m[Aggregator] Connection established with aggregator {addr}\033[0m")
            buffer = ""

            while True:
                try:
                    data = conn.recv(1024)
                    if not data:
                        print("\033[94m[Aggregator] Aggregator disconnected\033[0m")
                        break
                    buffer += data.decode()
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        try:
                            msg = json.loads(line.strip())
                            with aggregator_lock:
                                aggregator_data = msg
                                aggregator_timestamp = time.time()

                                # Inject aggregator probability into confidence history
                                probability = aggregator_data.get("probability", 0)
                                if probability > 0:
                                    with buffer_lock:
                                        confidence_history.append(probability)
                                        if len(confidence_history) > CONF_HISTORY_LEN:
                                            confidence_history.pop(0)

                        except json.JSONDecodeError:
                            print(f"\033[91m[Aggregator] Invalid JSON message: {line.strip()}\033[0m")
                except ConnectionResetError:
                    print("\033[91m[Aggregator] Connection lost by aggregator. Waiting for reconnection\033[0m")
                    break
                except Exception as e:
                    print(f"\033[91m[Aggregator] Unexpected error: {e}\033[0m")
                    break
        except Exception as e:
            print(f"\033[91m[Aggregator] Error accepting connection: {e}\033[0m")
            time.sleep(2)


####################################################################################
####################                 AoA THREAD                 ####################
####################################################################################
def angle_loop():
    global latest_angle
    try:
        dll_path = os.path.join(os.getcwd(), "libusb-1.0.dll")
        backend = usb.backend.libusb1.get_backend(find_library=lambda x: dll_path)
        if not backend:
            print("\033[91m[AoA] libusb backend failed to load.\033[0m")
            return

        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018, backend=backend)
        if not dev:
            print("\033[91m[AoA] ReSpeaker not found.\033[0m")
            return

        mic = Tuning(dev)
        print("\033[96m[AoA] Tracking started...\033[0m")

        while True:
            try:
                angle = mic.direction
                with angle_lock:
                    latest_angle = angle
                time.sleep(1)
            except Exception as e:
                print(f"\033[91m[AoA] Error reading angle: {e} \n\033[0m")
                time.sleep(1)
    finally:
        try:
            mic.close()
        except:
            pass
        print("\033[96m[AoA] USB resources released.\033[0m")


####################################################################################
####################          FLASK URL VISUALIZATION           ####################
####################################################################################                
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      background-color: black;
      overflow: hidden;
      height: 100%;
      width: 100%;
    }
    #wrapper {
      position: absolute;
      top: 0;
      left: 0;
    }
    img {
      display: block;
      margin: 0;
      padding: 0;
      width: auto;
      height: auto;
    }
  </style>
</head>
<body>
  <div id="wrapper">
    <img src="{{ url_for('spectrogram') }}">
  </div>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/spectrogram')
def spectrogram():
    return Response(generate_spectrogram(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_spectrogram():
    global audio_buffer, central_node_prediction_label, central_node_probability, confidence_history

    while True:
        with buffer_lock:
            y = audio_buffer.copy()
            conf_hist = confidence_history.copy()

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 5.625),
            gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.04}
        )

        black_color = "#0a0a0a"
        ax1.set_facecolor(black_color)
        dark_purple = '#030313'
        ax2.set_facecolor(dark_purple)
        fig.patch.set_facecolor('white')

        # Top: Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=SAMPLE_RATE, x_axis='time', y_axis='log', ax=ax1)
        ax1.set_ylabel('')
        ax1.set_xlabel('')
        ax1.tick_params(axis='x', bottom=False, labelbottom=False)
        ax1.set_yticks([])
        ax1.text(0.01, 0.1, '500 Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.3, '1k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.55, '2k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)
        ax1.text(0.01, 0.8, '4k Hz', color='white', ha='left', va='center', fontsize=10, transform=ax1.transAxes)

        # Detection check
        with aggregator_lock:
            aggregator_drone = (time.time() - aggregator_timestamp < 2) and (aggregator_data.get("probability", 0) > 0.7)
            aggregator_probability = aggregator_data.get("probability", 0)
            aggregator_node = aggregator_data.get("node", "Unknown")
            aggregator_direction = aggregator_data.get("direction", "Unknown")

        drone_detected = (central_node_prediction_label == "Drone") or aggregator_drone

        if drone_detected:
            if central_node_prediction_label == "Drone" and not aggregator_drone:
                source = "[Central Node]"
                displayed_probability = central_node_probability
            elif aggregator_drone:
                source = f"[{aggregator_node}] [Direction: {aggregator_direction}]"
                displayed_probability = aggregator_probability
            else:
                source = "[Unknown]"
                displayed_probability = 0.0

            ax1.text(0.5, 0.52, "DRONE DETECTED", transform=ax1.transAxes,
                     fontsize=32, color='red', ha='center', va='center_baseline', fontweight='bold')
            ax1.text(0.5, 0.43, f"[ {displayed_probability:.2f} ] {source}", transform=ax1.transAxes,
                     fontsize=20, color='red', ha='center', va='center_baseline', fontweight='bold')
            rect = Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                             fill=False, color='red', linewidth=10)
            ax1.add_patch(rect)

        # Bottom: Confidence plot
        conf_array = np.array(conf_hist)

        # --- Inject the latest active probability ---
        if drone_detected:
            conf_array = np.append(conf_array, displayed_probability)
            if len(conf_array) > CONF_HISTORY_LEN:
                conf_array = conf_array[-CONF_HISTORY_LEN:]

        conf_array = np.convolve(conf_array, np.ones(3)/3, mode='same')
        x_vals = np.arange(len(conf_array))
        color_for_dot = "red"
        ax2.plot(x_vals, conf_array, color=color_for_dot, linewidth=2)
        light_orange = '#ff9465'
        ax2.fill_between(x_vals, 0, conf_array, color=light_orange, alpha=0.7)
        ax2.set_ylim([0, 1])
        ax2.set_xlim([0, CONF_HISTORY_LEN - 1])
        ax2.set_yticks([])

        ax2.text(0.35, 0.15, 'Background', color='white', ha='left', va='center', fontsize=10)
        ax2.text(0.35, 0.85, 'Drone', color='white', ha='left', va='center', fontsize=10)

        ax2.tick_params(axis='x', bottom=False, labelbottom=False)
        ax2.grid(True, linestyle='--', alpha=0.0, color='white')

        for spine in ax2.spines.values():
            spine.set_visible(False)

        buf = io.BytesIO()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buf.read() + b'\r\n')
        time.sleep(STEP_SIZE / SAMPLE_RATE)



####################################################################################
####################                    MAIN                    ####################
#################################################################################### 
if __name__ == '__main__':
    threading.Thread(target=angle_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=aggregator_listener, daemon=True).start()

    if ENABLE_VISUALIZATION:
        print("\033[94m[Visualization] Visit http://localhost:5000 to see the live spectrogram.\n\033[0m")
        app.run(host='0.0.0.0', port=XXXX, threaded=True)
    else:
        while True:
            time.sleep(1)
