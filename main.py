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

# Global variables with thread-safe initialization
target_macs = []
target_macs_set = set()
latest_rssi = {}
last_update_time = {}
training_requests = {}
prediction_active = {}
prediction_stop_events = {}
lock = threading.Lock()
model_lock = threading.Lock()
training_lock = threading.Lock()

# Global models and scalers
global_models = {}
global_scalers = {}
training_in_progress = set()
training_semaphore = threading.Semaphore(2)  # Limit concurrent trainings

def validate_mac(mac):
    mac_pattern = re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$')
    return bool(mac_pattern.match(mac))

def refresh_target_macs():
    global target_macs, target_macs_set, latest_rssi, last_update_time
    with lock:
        devices = get_all_devices()
        new_target_macs = [device[1].lower() for device in devices if device[5]]
        current_macs = set(latest_rssi.keys())
        new_macs = set(new_target_macs)
        
        for mac in new_macs - current_macs:
            latest_rssi[mac] = {1: deque(maxlen=2), 2: deque(maxlen=2), 3: deque(maxlen=2), 4: deque(maxlen=2)}
            last_update_time[mac] = {1: 0, 2: 0, 3: 0, 4: 0}
            prediction_active[mac] = False
            prediction_stop_events[mac] = threading.Event()
            training_requests[mac] = False
        
        for mac in current_macs - new_macs:
            if mac in prediction_active:
                prediction_stop_events[mac].set()
            for collection in [latest_rssi, last_update_time, prediction_active, 
                             prediction_stop_events, training_requests]:
                if mac in collection:
                    del collection[mac]
        
        target_macs = new_target_macs
        target_macs_set = set(target_macs)
        
        # Cleanup global models
        with model_lock:
            current_macs_lower = set(mac.lower() for mac in new_target_macs)
            for mac in list(global_models.keys()):
                if mac not in current_macs_lower:
                    del global_models[mac]
                    if mac in global_scalers:
                        del global_scalers[mac]

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    for topic in topics:
        client.subscribe(topic, qos=0)

def on_message(client, userdata, msg, properties=None):
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
    message = {"mac": mac, "progress": round(progress, 2)}
    client.publish(training_status_topic, json.dumps(message), qos=0)
    print(f"Published progress for MAC {mac}: {message}")

def collect_data(mac, label, num_samples=500, interval=0.1, progress_callback=None):
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
                    rssi_values.append(values[-1] if len(values) == 1 else values[-1] * 0.7 + values[-2] * 0.3)
                else:
                    rssi_values.append(-100)
        
        if all(v != -100 for v in rssi_values):
            data.append(rssi_values + [label])
            print(f"MAC {mac}: RSSI values: {rssi_values} for label {label}")
        else:
            print(f"MAC {mac}: Missing RSSI data, skipping sample")
        
        if progress_callback and i % 10 == 0:
            progress_callback(i / num_samples)
    
    if progress_callback:
        progress_callback(1.0)
    return data

def simulate_not_in_position_data(in_position_data, num_samples=500, noise_std=10.0, shift_range=100.0):
    if not in_position_data:
        return []
    
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
    mac_lower = mac.lower()
    if mac_lower not in target_macs_set:
        print(f"MAC {mac} not in target devices. Skipping training.")
        publish_progress(mac, 100)
        return
    
    total_phases = 3
    completed_phases = 0
    
    def progress_callback(phase_progress):
        overall_progress = (completed_phases + phase_progress) / total_phases * 100
        publish_progress(mac, overall_progress)
    
    print(f"Place device {mac} in its locked position.")
    locked_data = collect_data(mac, 1, progress_callback=lambda p: progress_callback(p * 0.5))
    completed_phases += 1
    
    print(f"Simulating 'not in locked position' data for MAC {mac}")
    not_locked_data = simulate_not_in_position_data(locked_data)
    completed_phases += 1
    
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
    
    # Class weighting for imbalanced data
    class_weights = {0: 1.0, 1: len(not_locked_data)/len(locked_data)} if locked_data and not_locked_data else None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Improved model architecture
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])
    
    # Custom callback for training progress
    class TrainingProgressCallback(keras.callbacks.Callback):
        def __init__(self, mac, completed_phases, total_phases):
            self.mac = mac
            self.completed_phases = completed_phases
            self.total_phases = total_phases
            self.max_epochs = 100
        
        def on_epoch_end(self, epoch, logs=None):
            phase_progress = min((epoch + 1) / self.max_epochs, 1.0)
            overall_progress = (self.completed_phases + phase_progress) / self.total_phases * 100
            publish_progress(self.mac, overall_progress)
    
    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            TrainingProgressCallback(mac, completed_phases, total_phases),
            early_stopping
        ],
        class_weight=class_weights,
        verbose=0
    )
    
    model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
    model.save(model_file)
    
    # Update global models immediately
    with model_lock:
        global_models[mac_lower] = model
        global_scalers[mac_lower] = scaler
    
    print(f"Model and scaler saved for MAC {mac}")
    
    # Evaluate model performance
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Model evaluation for {mac}: "
          f"Accuracy={test_acc:.4f}, "
          f"Precision={test_precision:.4f}, "
          f"Recall={test_recall:.4f}")
    
    publish_progress(mac, 100)

def publish_prediction(mac, status, confidence, rssi_values):
    message = {
        "mac": mac,
        "status": status,
        "confidence": float(confidence),
        "rssi_values": rssi_values,
        "timestamp": time.time()
    }
    client.publish(publish_topic, json.dumps(message), qos=0)
    print(f"Published: {message}")

def predict():
    refresh_target_macs()
    counter = 0
    refresh_interval = 600  # Refresh every 10 minutes
    
    while True:
        if counter % refresh_interval == 0:
            refresh_target_macs()
            # Load new models
            with model_lock:
                for mac in target_macs:
                    mac_lower = mac.lower()
                    if mac_lower in global_models:
                        continue
                    model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
                    scaler_file = f'scaler_{mac.replace(":", "_")}.joblib'
                    if os.path.exists(model_file) and os.path.exists(scaler_file):
                        try:
                            global_models[mac_lower] = keras.models.load_model(model_file)
                            global_scalers[mac_lower] = joblib.load(scaler_file)
                            prediction_active[mac_lower] = True
                            print(f"Loaded model and scaler for MAC {mac}")
                        except Exception as e:
                            print(f"Error loading model for MAC {mac}: {e}")
                            prediction_active[mac_lower] = False
        
        counter += 1
        
        current_time = time.time()
        with lock:
            active_macs = [mac for mac in target_macs 
                          if prediction_active.get(mac.lower(), False)
                          and not prediction_stop_events[mac.lower()].is_set()]
        
        for mac in active_macs:
            mac_lower = mac.lower()
            rssi_values = []
            
            with lock:
                for esp in [1, 2, 3, 4]:
                    if latest_rssi[mac_lower][esp] and (current_time - last_update_time[mac_lower][esp]) < 5.0:
                        values = list(latest_rssi[mac_lower][esp])
                        rssi_values.append(values[-1] if len(values) == 1 else values[-1] * 0.7 + values[-2] * 0.3)
                    else:
                        rssi_values.append(-100)
            
            if all(v != -100 for v in rssi_values):
                try:
                    with model_lock:
                        if mac_lower not in global_models or mac_lower not in global_scalers:
                            continue
                        scaler = global_scalers[mac_lower]
                        model = global_models[mac_lower]
                    
                    X = scaler.transform([rssi_values])
                    prediction = model.predict(X, verbose=0)[0][0]
                    status = "locked" if prediction > 0.5 else "not locked"
                    confidence = prediction if prediction > 0.5 else 1 - prediction
                    publish_prediction(mac, status, confidence, rssi_values)
                except Exception as e:
                    print(f"Prediction error for MAC {mac}: {e}")
        
        time.sleep(0.1)

@app.route('/train/<mac>', methods=['GET'])
def request_training(mac):
    mac = mac.lower()
    if not validate_mac(mac):
        return jsonify({"error": "Invalid MAC address format"}), 400
    
    with lock:
        if mac not in target_macs_set:
            return jsonify({"error": f"MAC {mac} not found in target devices"}), 404
        
        training_requests[mac] = True
    
    return jsonify({"message": f"Training requested for MAC {mac}"}), 200

@app.route('/refresh_devices', methods=['GET'])
def refresh_devices():
    refresh_target_macs()
    with lock:
        return jsonify({
            "message": "Devices refreshed successfully",
            "target_macs": list(target_macs),
            "training_available": {mac: os.path.exists(f'rtls_model_{mac.replace(":", "_")}.keras') 
                                for mac in target_macs}
        }), 200

def training_worker(mac):
    """Worker function for training in a separate thread."""
    training_semaphore.acquire()
    try:
        # Stop prediction for this device during training
        with lock:
            prediction_stop_events[mac].set()
            prediction_active[mac] = False
        
        print(f"Starting training for MAC {mac}")
        train_model(mac)
        print(f"Training completed for MAC {mac}")
        
    except Exception as e:
        print(f"Training failed for MAC {mac}: {e}")
    finally:
        # Re-enable prediction for this device
        with lock:
            prediction_stop_events[mac].clear()
            prediction_active[mac] = True
        
        with training_lock:
            training_in_progress.discard(mac)
        
        training_semaphore.release()

def process_training_requests():
    """Process training requests in parallel."""
    while True:
        with lock:
            training_macs = [mac for mac, req in training_requests.items() if req]
        
        with training_lock:
            for mac in training_macs:
                if mac in training_in_progress:
                    continue
                
                training_requests[mac] = False
                training_in_progress.add(mac)
                
                # Start training in a new thread
                t = threading.Thread(target=training_worker, args=(mac,))
                t.daemon = True
                t.start()
        
        time.sleep(1)

def main():
    global client
    client = mqtt.Client(client_id="rtls_server", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    
    # Configure MQTT client
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker, port)
        client.loop_start()
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    
    # Initialize target devices
    refresh_target_macs()
    
    # Pre-load existing models
    with model_lock:
        for mac in target_macs:
            mac_lower = mac.lower()
            model_file = f'rtls_model_{mac.replace(":", "_")}.keras'
            scaler_file = f'scaler_{mac.replace(":", "_")}.joblib'
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                try:
                    global_models[mac_lower] = keras.models.load_model(model_file)
                    global_scalers[mac_lower] = joblib.load(scaler_file)
                    prediction_active[mac_lower] = True
                    print(f"Pre-loaded model for MAC {mac}")
                except Exception as e:
                    print(f"Error pre-loading model for MAC {mac}: {e}")
    
    # Start threads
    prediction_thread = threading.Thread(target=predict, daemon=True)
    prediction_thread.start()
    
    training_thread = threading.Thread(target=process_training_requests, daemon=True)
    training_thread.start()
    
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False),
        daemon=True
    )
    flask_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        with lock:
            for mac in target_macs:
                prediction_stop_events[mac].set()
                prediction_active[mac] = False
        client.loop_stop()
        client.disconnect()
        sys.exit(0)

if __name__ == "__main__":
    main()