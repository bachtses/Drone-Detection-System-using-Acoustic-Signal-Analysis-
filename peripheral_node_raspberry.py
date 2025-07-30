import os
import numpy as np
import librosa
import tflite_runtime.interpreter as tflite
import pyaudio
import time
import threading
import json
import usb.core
import usb.util
import usb.backend.libusb1
from tuning import Tuning
from lora import LoRaTransmitter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

####################################################################################
####################                  CONSTANTS                 ####################
####################################################################################
node_id = 1

SLIDING_SPEED = 1
CONF_HISTORY_LEN = 50
SAMPLE_RATE = 16000
AUDIO_LENGTH = 2
N_MFCC = 40
CHUNK = 1024
STEP_SIZE = int(SLIDING_SPEED * SAMPLE_RATE)
audio_buffer = np.zeros(SAMPLE_RATE * AUDIO_LENGTH, dtype=np.float32)
buffer_lock = threading.Lock()
latest_probability = 0
confidence_history = []
latest_angle = -1
angle_lock = threading.Lock()

####################################################################################
####################                LOAD MODEL                  ####################
####################################################################################
try:
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    print("\033[96m[Model] TFLite model loaded successfully.\033[0m")
except Exception as e:
    print("\033[91m[Model] Failed to load the TFLite model:\033[0m", e)
    exit(1)

####################################################################################
####################          INITIALIZE AUDIO STREAM           ####################
####################################################################################
p = pyaudio.PyAudio()
device_index = None
respeaker_keyword = "respeaker"
print("\033[96m[Microphone] Checking for ReSpeaker microphone device:\033[0m")
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    name_lower = dev_info['name'].lower()
    if dev_info['maxInputChannels'] > 0 and respeaker_keyword in name_lower:
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
print("\033[96m[Microphone] Audio stream opened. Listening for real-time audio...\033[0m")


####################################################################################
####################             FEATURE EXTRACTION             ####################
####################################################################################
def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)  # shape: (40, time_steps)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)                # normalize entire matrix
    return mfcc.T           

####################################################################################
####################             LORA SENDER INIT               ####################
####################################################################################
lora = LoRaTransmitter()


####################################################################################
####################                 PREDICTIONS                ####################
####################################################################################
def prediction_loop():
    global audio_buffer, latest_probability, latest_angle

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        try:
            new_frames = []
            num_new_chunks = int(STEP_SIZE / CHUNK)
            for _ in range(num_new_chunks):
                data = stream.read(CHUNK, exception_on_overflow=False)
                new_frames.append(np.frombuffer(data, dtype=np.float32))
            new_audio = np.concatenate(new_frames)

            with buffer_lock:
                audio_buffer = np.roll(audio_buffer, -len(new_audio))
                audio_buffer[-len(new_audio):] = new_audio

            features = extract_mfcc_features(audio_buffer, SAMPLE_RATE)
            features = features[np.newaxis, :, :].astype(np.float32) 

            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            prediction = output_data[0][0]
            label = 1 if prediction > 0.7 else 0  # 1 = Drone, 0 = Background
            latest_probability = prediction

            confidence_history.append(latest_probability)
            if len(confidence_history) > CONF_HISTORY_LEN:
                confidence_history.pop(0)

            with angle_lock:
                current_angle = latest_angle

            print(f"[Node {node_id}] Prediction: {label} | Probability: {prediction:.3f} | AoA: {current_angle}°")

            message = json.dumps({
                "id": node_id,
                "l": label,
                "p": round(float(latest_probability), 3),
                "a": current_angle,
                "s": None
            })

            #Total load/bytes of message via LoRa
            #print(f"\033[93m[DEBUG] Message size: {len(message.encode('utf-8'))} bytes\033[0m")

            # Offload LoRa transmission to avoid blocking
            if label == 1:
                threading.Thread(target=lora.send_message, args=(message,), daemon=True).start()

        except Exception as e:
            print(f"\033[91m[PREDICTION LOOP ERROR] {e}\033[0m")
            time.sleep(1)  



####################################################################################
####################                 AoA THREAD                 ####################
####################################################################################
def angle_loop():
    global latest_angle
    try:
        backend = usb.backend.libusb1.get_backend()
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
                print(f"\033[91m[AoA] Error reading angle: {e}\033[0m")
                time.sleep(1)
    finally:
        try:
            mic.close()
        except:
            pass
        print("\033[96m[AoA] USB resources released.\033[0m")


####################################################################################
####################            TEMPERATURE MONITOR             ####################
####################################################################################
def monitor_temperature():
    while True:
        temp_output = os.popen("vcgencmd measure_temp").read()
        try:
            temp_value = float(temp_output.replace("temp=", "").replace("'C\n", ""))
            if temp_value >= 75:
                print(f"\033[91m[Warning] High temperature detected: {temp_value}°C\033[0m")
        except ValueError:
            print("[Temp Monitor] Failed to read temperature.")
        time.sleep(10)  


####################################################################################
####################                    MAIN                    ####################
####################################################################################
if __name__ == '__main__':
    time.sleep(2)
    try:
        lora.send_message(f"[Aggregator] [Node:{node_id}] Online")
    except Exception as e:
        print(f"\033[91m[Aggregator] Failed to send boot message: {e}\033[0m")
    
    threading.Thread(target=angle_loop, daemon=True).start()
    threading.Thread(target=prediction_loop, daemon=True).start()
    threading.Thread(target=monitor_temperature, daemon=True).start()



    while True:
        time.sleep(1)
