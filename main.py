import paho.mqtt.client as mqtt
import json
import time
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import sys
from collections import deque
import os
import threading
from database import get_all_devices
from flask import Flask, jsonify
import re

# MQTT settings
broker = 'security.local'
port = 1883
topics = ['esp32_1/rssi', 'esp32_2/rssi', 'esp32_3/rssi', 'esp32_4/rssi']
publish_topic = 'rtls/position_status'
training_status_topic = 'training/status'

# Flask app setup
app = Flask(__name__)

# Global variables
target_macs = []
target_macs_set = set()
latest_rssi = {}
last_update_time = {}
training_requests = {}  # Dictionary to track training requests per MAC
prediction_active = {}  # Dictionary to track prediction status per MAC
prediction_stop_events = {}  # Dictionary to track stop events per MAC
lock = threading.Lock()  # Thread lock for shared resources

def validate_mac(mac):
    """Validate MAC address format."""
    mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
    return bool(mac_pattern.match(mac))

def refresh_target_macs():
    """Refresh the list of target MACs from the database."""
    global target_macs, target_macs_set, latest_rssi, last_update_time
    with lock:
        devices = get_all_devices()
        new_target_macs = [device[1].lower() for device in devices if device[0] == True]
        current_macs = set(latest_rssi.keys())
        new_macs = set(new_target_macs)
        
        for mac in new_macs - current_macs:
            latest_rssi[mac] = {1: deque(maxlen=2), 2: deque(maxlen=2), 3: deque(maxlen=2), 4: deque(maxlen=2)}
            last_update_time[mac] = {1: 0, 2: 0, 3: 0, 4: 0}
            prediction_active[mac] = False
            prediction_stop_events[mac] = threading.Event()
        
        for mac in current_macs - new_macs:
            if mac in prediction_active:
                prediction_stop_events[mac].set()
            del latest_rssi[mac]
            del last_update_time[mac]
            del prediction_active[mac]
            del prediction_stop_events[mac]
        
        target_macs = new_target_macs
        target_macs_set = set(target_macs)

def on_connect(client, userdata, flags, reason_code, properties):
    """Handle MQTT connection."""
    print(f"Connected with result code {reason_code}")
    for topic in topics:
        client.subscribe(topic, qos=0)

def on_message(client, userdata, msg, properties=None):
    """Handle incoming MQTT messages."""
    global training_requests
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        mac = data['mac'].lower()
        if mac in target_macs_set:
            esp = data['esp']
            rssi = data['rssi']
            with lock:
                latest_rssi[mac][esp].append(rssi)
                last_update_time[mac][esp] = time.time()
            print(f"Received RSSI for MAC {mac}: ESP{esp} = {rssi}")
    except Exception as e:
        print(f"Error processing message: {e}")

def publish_progress(mac, progress):
    """Publish the training progress percentage for a MAC."""
    message = {"mac": mac, "progress": round(progress, 2)}
    client.publish(training_status_topic, json.dumps(message), qos=0)
    print(f"Published progress for MAC {mac}: {message}")

def collect_data(mac, label, num_samples=500, interval=0.1, progress_callback=None):
    """Collect RSSI data for a specific MAC."""
    data = []
    mac_lower = mac.lower()
    print(f"Collecting {num_samples} samples for MAC {mac} with label {label}")
    for i in range(num_samples):
        time.sleep(interval)
        current_time = time.time()
        rssi_values = []
        with lock:
            for esp in [1, 2, 3, 4]:
                if latest_rssi[mac_lower][esp] and (current_time - last_update_time[mac_lower][esp]) < 5.0:
                    values = list(latest_rssi[mac_lower][esp])
                    rssi_values.append(values[0] if len(values) == 1 else values[-1] * 0.7 + values[-2] * 0.3)
                else:
                    rssi_values.append(-100)
        print(f"MAC {mac}: RSSI values: {rssi_values} for label {label}")
        if all(v != -100 for v in rssi_values):
            data.append(rssi_values + [label])
        else:
            print(f"MAC {mac}: Missing RSSI data, skipping sample")
        if progress_callback and i % 100 == 0:
            progress_callback(i / num_samples)
    if progress_callback:
        progress_callback(1.0)
    return data

def simulate_not_in_position_data(in_position_data, num_samples=500, noise_std=8.0, shift_range=20.0):
    """Simulate 'not in position' data."""
    data = []
    in_position_rssi = np.array([d[:4] for d in in_position_data], dtype=float)
    for i in range(num_samples):
        base_rssi = in_position_rssi[np.random.randint(len(in_position_rssi))]
        shift_factor = (i / (num_samples - 1)) * shift_range
        directional_shift = np.full(4, shift_factor)
        noise = np.random.normal(0, noise_std, size=4)
        moved_rssi = base_rssi + directional_shift + noise
        moved_rssi = np.clip(moved_rssi, -100, -30)
        data.append(list(moved_rssi) + [0])
        if i % 100 == 0:
            print(f"Simulated RSSI sample {i}: {moved_rssi}")
    return data

def train_model(mac):
    """Train model for a specific MAC."""
    mac_lower = mac.lower()
    if mac_lower not in target_macs_set:
        print(f"MAC {mac} not in target devices. Skipping training.")
        publish_progress(mac, 100)
        return
    
    total_phases = 2
    completed_phases = 0
    
    def progress_callback(phase_progress):
        overall_progress = (completed_phases + phase_progress) / total_phases * 100
        publish_progress(mac, overall_progress)
    
    print(f"Place device {mac} in its locked position.")
    locked_data = collect_data(mac, 1, progress_callback=progress_callback)
    completed_phases += 1
    
    print(f"Simulating 'not in locked position' data for MAC {mac}")
    not_locked_data = simulate_not_in_position_data(locked_data)
    
    all_data = locked_data + not_locked_data
    if not all_data:
        print(f"No valid data for MAC {mac}. Skipping training.")
        publish_progress(mac, 100)
        return
    
    X = np.array([d[:4] for d in all_data], dtype=float)
    y = np.array([d[4] for d in all_data], dtype=int)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    scaler_file = f'scaler_{mac.replace(":", "_")}.joblib'
    joblib.dump(scaler, scaler_file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    class TrainingProgressCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
        
        def on_epoch_end(self, epoch, logs=None):
            phase_progress = (epoch + 1) / 100
            overall_progress = (completed_phases + phase_progress) / total_phases * 100
            publish_progress(mac, overall_progress)
    
    callback = TrainingProgressCallback()
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
              callbacks=[callback], verbose=0)
    model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
    model.save(model_file)
    print(f"Model and scaler saved for MAC {mac}")
    publish_progress(mac, 100)

def publish_prediction(mac, status, confidence, rssi_values):
    """Publish prediction results."""
    message = {
        "mac": mac,
        "status": status,
        "confidence": float(confidence),
    }
    client.publish(publish_topic, json.dumps(message), qos=0)
    print(f"Published: {message}")

def predict():
    """Run continuous prediction for all MACs."""
    refresh_target_macs()
    models = {}
    scalers = {}
    
    with lock:
        for mac in target_macs:
            mac_lower = mac.lower()
            model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
            scaler_file = f'scaler_{mac.replace(":", "_")}.joblib'
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                models[mac_lower] = keras.models.load_model(model_file)
                scalers[mac_lower] = joblib.load(scaler_file)
                prediction_active[mac_lower] = True
                print(f"Loaded model and scaler for MAC {mac}")
            else:
                print(f"Model or scaler missing for MAC {mac}")
    
    counter = 0
    refresh_interval = 600
    while True:
        if counter % refresh_interval == 0:
            refresh_target_macs()
        counter += 1
        
        current_time = time.time()
        for mac in list(target_macs):
            mac_lower = mac.lower()
            if mac_lower not in models or not prediction_active.get(mac_lower, False):
                continue
                
            if prediction_stop_events[mac_lower].is_set():
                continue
                
            with lock:
                rssi_values = []
                for esp in [1, 2, 3, 4]:
                    if latest_rssi[mac_lower][esp] and (current_time - last_update_time[mac_lower][esp]) < 5.0:
                        values = list(latest_rssi[mac_lower][esp])
                        rssi_values.append(values[-1] if len(values) == 1 else values[-1] * 0.7 + values[-2] * 0.3)
                    else:
                        rssi_values.append(-100)
            
            if all(v != -100 for v in rssi_values):
                X = scalers[mac_lower].transform([rssi_values])
                prediction = models[mac_lower].predict(X, verbose=0)[0][0]
                status = "locked" if prediction > 0.5 else "not locked"
                confidence = prediction if prediction > 0.5 else 1 - prediction
                publish_prediction(mac, status, confidence, rssi_values)
        
        time.sleep(0.1)

@app.route('/train/<mac>', methods=['POST'])
def request_training(mac):
    """API endpoint to request training for a specific MAC."""
    mac = mac.lower()
    if not validate_mac(mac):
        return jsonify({"error": "Invalid MAC address format"}), 400
    
    if mac not in target_macs_set:
        return jsonify({"error": f"MAC {mac} not found in target devices"}), 404
    
    with lock:
        training_requests[mac] = True
    
    return jsonify({"message": f"Training requested for MAC {mac}"}), 200

@app.route('/refresh_devices', methods=['POST'])
def refresh_devices():
    """API endpoint to refresh the list of target devices."""
    refresh_target_macs()
    with lock:
        mac_list = list(target_macs)
    return jsonify({"message": "Devices refreshed successfully", "target_macs": mac_list}), 200

def main():
    """Main control loop."""
    global client
    client = mqtt.Client(client_id="", callback_api_version=mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(broker, port)
        client.loop_start()
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    
    prediction_thread = threading.Thread(target=predict, daemon=True)
    prediction_thread.start()
    
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False), daemon=True)
    flask_thread.start()
    
    try:
        while True:
            with lock:
                for mac in list(training_requests.keys()):
                    if training_requests[mac]:
                        print(f"Processing training request for MAC {mac}")
                        training_requests[mac] = False
                        prediction_stop_events[mac].set()
                        with lock:
                            prediction_active[mac] = False
                        train_model(mac)
                        print(f"Training completed for MAC {mac}")
                        prediction_stop_events[mac].clear()
                        with lock:
                            prediction_active[mac] = True
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        with lock:
            for mac in target_macs:
                prediction_stop_events[mac].set()
                prediction_active[mac] = False
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()