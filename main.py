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

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MQTT settings
broker = 'security.local'
port = 1883
topics = ['esp32_1/rssi', 'esp32_2/rssi', 'esp32_3/rssi', 'esp32_4/rssi', 'mode/train']
publish_topic = 'rtls/position_status'
training_status_topic = 'training/status'

# Global variables
target_macs = []
target_macs_set = set()
latest_rssi = {}
last_update_time = {}
training_requested = False
prediction_active = False
prediction_stop_event = threading.Event()

def refresh_target_macs():
    global target_macs, target_macs_set, latest_rssi, last_update_time
    devices = get_all_devices()
    new_target_macs = [device[1].lower() for device in devices if device[5] == True]
    current_macs = set(latest_rssi.keys())
    new_macs = set(new_target_macs)
    
    # Add new MACs
    for mac in new_macs - current_macs:
        latest_rssi[mac] = {1: deque(maxlen=2), 2: deque(maxlen=2), 3: deque(maxlen=2), 4: deque(maxlen=2)}
        last_update_time[mac] = {1: 0, 2: 0, 3: 0, 4: 0}
    
    # Remove old MACs
    for mac in current_macs - new_macs:
        del latest_rssi[mac]
        del last_update_time[mac]
    
    target_macs = new_target_macs
    target_macs_set = set(target_macs)

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    for topic in topics:
        client.subscribe(topic, qos=0)

def on_message(client, userdata, msg, properties=None):
    global training_requested
    try:
        if msg.topic == 'mode/train':
            print("Training mode requested via MQTT")
            training_requested = True
        else:
            data = json.loads(msg.payload.decode('utf-8'))
            mac = data['mac'].lower()
            if mac in target_macs_set:
                esp = data['esp']
                rssi = data['rssi']
                latest_rssi[mac][esp].append(rssi)
                last_update_time[mac][esp] = time.time()
                print(f"Received RSSI for MAC {mac}: ESP{esp} = {rssi}")
    except Exception as e:
        print(f"Error processing message: {e}")

def publish_progress(progress):
    """Publish the training progress percentage to 'training/status'."""
    message = {"progress": round(progress, 2)}
    client.publish(training_status_topic, json.dumps(message), qos=0)
    print(f"Published progress: {message}")

def collect_data(label, num_samples=500, interval=0.1, progress_callback=None):
    """Collect RSSI data for all target MACs with progress updates."""
    data = {mac: [] for mac in target_macs}
    print(f"Collecting {num_samples} samples for all MACs with label {label}")
    for _ in range(num_samples):
        time.sleep(interval)
        current_time = time.time()
        for mac in target_macs:
            mac_lower = mac.lower()
            rssi_values = []
            for esp in [1, 2, 3, 4]:
                if latest_rssi[mac_lower][esp] and (current_time - last_update_time[mac_lower][esp]) < 5.0:
                    values = list(latest_rssi[mac_lower][esp])
                    rssi_values.append(values[0] if len(values) == 1 else values[-1] * 0.7 + values[-2] * 0.3)
                else:
                    rssi_values.append(-100)
            print(f"MAC {mac}: RSSI values: {rssi_values} for label {label}")
            if all(v != -100 for v in rssi_values):
                data[mac].append(rssi_values + [label])
            else:
                print(f"MAC {mac}: Missing RSSI data, skipping sample")
        # Update progress every 100 iterations
        if progress_callback and _ % 100 == 0:
            progress_callback(_ / num_samples)
    if progress_callback:
        progress_callback(1.0)  # Signal completion of data collection
    return data

def simulate_not_in_position_data(in_position_data, num_samples=500, noise_std=8.0, shift_range=20.0):
    """Simulate 'not in position' data based on in-position data."""
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
        if i % 1000 == 0:
            print(f"Simulated RSSI sample {i}: {moved_rssi}")
    return data

def train_model():
    """Train models for each MAC with overall progress tracking."""
    refresh_target_macs()
    if not target_macs:
        print("No devices to train. Exiting.")
        return
    
    # Define total phases: 1 for data collection + 1 per MAC for training
    total_phases = 1 + len(target_macs)
    completed_phases = 0
    
    # Progress callback for data collection
    def data_collection_progress(phase_progress):
        overall_progress = (completed_phases + phase_progress) / total_phases * 100
        publish_progress(overall_progress)
    
    print("Place all devices in their locked positions.")
    locked_data = collect_data(1, progress_callback=data_collection_progress)
    completed_phases += 1  # Data collection phase completed
    
    print("Simulating 'not in locked position' data for all MACs")
    not_locked_data = {mac: simulate_not_in_position_data(locked_data[mac]) for mac in target_macs}
    
    # Training phase for each MAC
    for mac in target_macs:
        print(f"Training model for MAC {mac}")
        all_data = locked_data[mac] + not_locked_data[mac]
        if not all_data:
            print(f"No valid data for MAC {mac}. Skipping training.")
            completed_phases += 1
            continue
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
        
        # Custom callback to track training progress
        class TrainingProgressCallback(keras.callbacks.Callback):
            def __init__(self, completed_phases, total_phases):
                super().__init__()
                self.completed_phases = completed_phases
                self.total_phases = total_phases
            
            def on_epoch_end(self, epoch, logs=None):
                phase_progress = (epoch + 1) / 100  # Assuming 5000 epochs
                overall_progress = (self.completed_phases + phase_progress) / self.total_phases * 100
                publish_progress(overall_progress)
        
        callback = TrainingProgressCallback(completed_phases, total_phases)
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
                  callbacks=[callback], verbose=0)
        model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
        model.save(model_file)
        print(f"Model and scaler saved for MAC {mac}")
        completed_phases += 1  # Training phase for this MAC completed
    
    # Publish 100% progress when all training is complete
    publish_progress(100)

def publish_prediction(mac, status, confidence, rssi_values):
    """Publish prediction results to 'rtls/position_status'."""
    message = {
        "mac": mac,
        "status": status,
        "confidence": float(confidence),
    }
    client.publish(publish_topic, json.dumps(message), qos=0)
    print(f"Published: {message}")

def predict():
    """Run continuous prediction for all MACs."""
    global prediction_active
    prediction_stop_event.clear()
    prediction_active = True
    
    refresh_target_macs()
    models = {}
    scalers = {}
    
    for mac in target_macs:
        mac_lower = mac.lower()
        model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
        scaler_file = f'scaler_{mac.replace(":", "_")}.joblib'
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            models[mac_lower] = keras.models.load_model(model_file)
            scalers[mac_lower] = joblib.load(scaler_file)
            print(f"Loaded model and scaler for MAC {mac}")
        else:
            print(f"Model or scaler missing for MAC {mac}")
    
    counter = 0
    refresh_interval = 600  # Refresh every 60 seconds
    while not prediction_stop_event.is_set():
        if counter % refresh_interval == 0:
            refresh_target_macs()
        counter += 1
        
        prediction_stop_event.wait(timeout=0.1)
        if prediction_stop_event.is_set():
            break
            
        current_time = time.time()
        for mac in target_macs:
            mac_lower = mac.lower()
            if mac_lower not in models:
                continue
                
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
    
    prediction_active = False
    print("Prediction stopped")

if __name__ == "__main__":
    # Start MQTT client
    client = mqtt.Client(client_id="", callback_api_version=mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5)
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(broker, port)
        client.loop_start()
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)

    # Main control loop
    try:
        while True:
            if not prediction_active:
                print("Starting prediction")
                prediction_thread = threading.Thread(target=predict)
                prediction_thread.start()
            
            if training_requested:
                print("Stopping prediction for training")
                training_requested = False
                prediction_stop_event.set()
                
                while prediction_active:
                    time.sleep(0.1)
                
                train_model()
                print("Training completed")
                
                prediction_stop_event.clear()
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        prediction_stop_event.set()
        print("Exiting...")
        client.loop_stop()
        client.disconnect()