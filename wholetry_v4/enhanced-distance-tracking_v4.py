#!/usr/bin/env python3
import socket
import threading
import json
import csv
import time
import os
from datetime import datetime
import math
import numpy as np

# Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000  # Port to use
DATA_DIR = "positions_data"  # Directory for data

# Map of each anchor's position (must match the anchor configuration)
'''
ANCHOR_POSITIONS = { #try
    'anchor1': (0, 0),
    'anchor2': (10, 0),
    'anchor3': (10, 10),
    'anchor4': (0, 10)
}'''
ANCHOR_POSITIONS = { #EP2
    'anchor1': (0, 0),
    'anchor2': (0, 7),
    'anchor3': (3.5, 7),
    'anchor4': (3.5, 0)
}
'''
ANCHOR_POSITIONS = { #home, livingroom
    'anchor1': (0, 0),
    'anchor2': (7, 0),
    'anchor3': (6, 4.3),
    'anchor4': (0.7, 6)
}'''


# Create data directory with timestamp
def create_data_directory(patient_id, patient_data):
    """
    Crea una cartella per i dati e salva anche il file info.json con i dati del paziente.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"positions_data_{patient_id}_{timestamp}"
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created data directory: {folder_name}")
    
    # Salva i dati del paziente nella cartella
    try:
        with open(os.path.join(folder_name, "info.json"), "w") as f:
            json.dump(patient_data, f, indent=4)
    except Exception as e:
        print(f"Error saving info.json: {e}")
    
    # Scrivi il link al file corrente (come già facevi)
    with open("latest_data_folder.txt", 'w') as f:
        f.write(folder_name)
    
    return folder_name


# Initialize CSV file
def initialize_csv(folder_name):
    csv_path = os.path.join(folder_name, 'wearable_data.csv')
    
    headers = [
        'timestamp', 
        'estimated_x', 
        'estimated_y', 
        'steps', 
        'step_distance_m',
        'position_distance_m',
        'fused_distance_m',
        'total_step_distance_m',
        'total_position_distance_m',
        'total_fused_distance_m'
    ]
    
    # Add columns for each anchor
    for anchor_name in ANCHOR_POSITIONS.keys():
        headers.append(f'distance_{anchor_name}')
        headers.append(f'rssi_{anchor_name}')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    
    print(f"Initialized CSV file: {csv_path}")
    
    # Create link to most recent file
    with open("latest_data_folder.txt", 'w') as f:
        f.write(folder_name)
    
    return csv_path

# Trilateration function to estimate position from distances
def estimate_position(distances):
    """
    Estimate the (x, y) position of the target using anchor distances.
    Supports 2 or more anchors. Uses inverse-square weighted interpolation.
    """
    # Filtra solo ancore con distanza valida
    valid_anchors = {anchor: dist for anchor, dist in distances.items() if dist > 0}
    n = len(valid_anchors)

    if n < 2:
        return None, None  # Troppo poche ancore

    anchor_items = list(valid_anchors.items())

    # Caso: 2 ancore → interpolazione pesata
    if n == 2:
        (a1, d1), (a2, d2) = anchor_items
        if a1 not in ANCHOR_POSITIONS or a2 not in ANCHOR_POSITIONS:
            return None, None

        x1, y1 = ANCHOR_POSITIONS[a1]
        x2, y2 = ANCHOR_POSITIONS[a2]

        # Peso inverso al quadrato della distanza
        w1 = 1.0 / (d1 ** 2)
        w2 = 1.0 / (d2 ** 2)
        w_total = w1 + w2

        estimated_x = (x1 * w1 + x2 * w2) / w_total
        estimated_y = (y1 * w1 + y2 * w2) / w_total

        return round(estimated_x, 2), round(estimated_y, 2)

    # Caso: >=3 ancore → media pesata classica
    total_weight = 0
    x_weighted_sum = 0
    y_weighted_sum = 0

    for anchor, distance in anchor_items:
        if anchor in ANCHOR_POSITIONS:
            x, y = ANCHOR_POSITIONS[anchor]
            weight = 1.0 / (distance ** 2)
            x_weighted_sum += x * weight
            y_weighted_sum += y * weight
            total_weight += weight

    if total_weight > 0:
        estimated_x = x_weighted_sum / total_weight
        estimated_y = y_weighted_sum / total_weight
        return round(estimated_x, 2), round(estimated_y, 2)

    return None, None


# Calculate distance from steps using anthropometric formula
def steps_to_distance(steps, height_m, age_years, gender):
    """
    Calculate distance from steps using adaptive coefficient based on height, age, and gender.
    """
    def get_step_coefficient(height_m, age, gender):
        # Adapted version of the function you gave
        ranges = {
        #   (eta_min, eta_max): (altezza_min, altezza_max, coeff_min, coeff_max)
            (5, 10): {'male': (100, 140, 0.41, 0.56), 'female': (100, 140, 0.40, 0.54)}, 
            (11, 15): {'male': (140, 175, 0.56, 0.77), 'female': (140, 165, 0.54, 0.73)},
            (16, 25): {'male': (170, 190, 0.72, 0.87), 'female': (160, 180, 0.68, 0.83)},
            (26, 35): {'male': (165, 195, 0.77, 0.90), 'female': (155, 185, 0.73, 0.86)},
            (36, 45): {'male': (160, 195, 0.75, 0.88), 'female': (150, 185, 0.71, 0.84)},
            (46, 55): {'male': (155, 190, 0.72, 0.86), 'female': (145, 180, 0.69, 0.82)},
            (56, 65): {'male': (150, 185, 0.70, 0.84), 'female': (140, 175, 0.67, 0.80)},
            (66, 75): {'male': (145, 180, 0.67, 0.82), 'female': (135, 170, 0.64, 0.78)},
            (76, 85): {'male': (140, 175, 0.62, 0.77), 'female': (130, 165, 0.59, 0.73)},
        }
        gender = gender.lower()
        height_cm = height_m * 100

        for (min_age, max_age), data in ranges.items():
            if min_age <= age <= max_age and gender in data:
                hmin, hmax, cmin, cmax = data[gender]
                if hmax > hmin:
                    coeff = cmin + (height_cm - hmin) / (hmax - hmin) * (cmax - cmin)
                    return max(cmin, min(cmax, coeff))
                elif height_cm >= hmin:
                    return cmax
                else:
                    return cmin
        return 0.7  # fallback

    coeff = get_step_coefficient(height_m, age_years, gender)
    return steps * coeff


# Calculate distance between two points
def calculate_distance_between_points(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Apply Kalman filter to smooth position estimates
class SimpleKalmanFilter:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-3):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        
    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Measurement update
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate
        
        return self.posteri_estimate

class Kalman2D:
    def __init__(self, process_var=1e-3, measurement_var=1e-3):
        self.x = np.array([[0.0], [0.0]])  # state: position and velocity
        self.P = np.eye(2)
        self.F = np.array([[1, 0], [0, 1]])  # identity, no dynamics
        self.H = np.array([[1, 0]])
        self.R = np.array([[measurement_var]])
        self.Q = np.eye(2) * process_var

    def update(self, z):
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update
        y = np.array([[z]]) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x[0, 0]  # return the estimated position


# Fusion algorithm to combine step and position-based distances
def fuse_distance_measurements(step_distance, position_distance, confidence_step=0.9, confidence_position=0.1):
    """
    Combine step-based and position-based distance measurements
    
    Args:
        step_distance: Distance calculated from steps (in meters)
        position_distance: Distance calculated from position tracking (in meters)
        confidence_step: Confidence weight for step-based distance (0-1)
        confidence_position: Confidence weight for position-based distance (0-1)
        
    Returns:
        Fused distance estimate (in meters)
    """
    # Normalize confidence weights
    total_confidence = confidence_step + confidence_position
    if total_confidence == 0:
        return (step_distance + position_distance) / 2  # Simple average if both weights are 0
    
    norm_confidence_step = confidence_step / total_confidence
    norm_confidence_position = confidence_position / total_confidence
    
    # Calculate weighted average
    fused_distance = (step_distance * norm_confidence_step) + (position_distance * norm_confidence_position)
    
    return fused_distance

# Update CSV with latest data from all anchors
def update_csv(csv_path, latest_data, position_history, step_history, kalman_x, kalman_y, user_height, user_age, user_gender):
    # Only update if we have data from at least one anchor
    if not latest_data:
        return
    
    # Get distances for each anchor
    distances = {}
    for anchor_id, data in latest_data.items():
        distances[anchor_id] = data.get('distance', -1)
    
    # Filtro sulle distanze troppo alte
    max_anchor_distance = max(
        math.dist(p1, p2) for p1 in ANCHOR_POSITIONS.values() for p2 in ANCHOR_POSITIONS.values())
    distance_cutoff = max_anchor_distance * 1.5

    # Logging: verifica quali anchor verranno scartati
    for aid, d in distances.items():
        if d <= 0:
            continue  # ignora già invalidi
        if d >= distance_cutoff:
            print(f"[FILTERED] Anchor '{aid}' → distanza {d:.2f} m > soglia {distance_cutoff:.2f} m")

    # Filtra distanze valide
    distances = {
        aid: d for aid, d in distances.items()
        if d > 0 and d < distance_cutoff
    }

    if not distances:
        print(f"[FILTERED] Nessuna distanza valida (< {distance_cutoff:.2f} m), pacchetto ignorato.")
        return


    # Estimate position using trilateration
    estimated_x, estimated_y = estimate_position(distances)
    
    # Apply Kalman filter to smooth the estimated position
    if estimated_x is not None and estimated_y is not None:
        estimated_x = kalman_x.update(estimated_x)
        estimated_y = kalman_y.update(estimated_y)

    # Get the latest step count (use value from any anchor that has it)
    step_count = None
    for anchor_id, data in latest_data.items():
        if 'steps' in data and data['steps'] is not None:
            step_count = data['steps']
            break
    
    # Calculate incremental distances if we have valid data
    current_timestamp = datetime.now()
    step_distance_m = 0
    position_distance_m = 0
    fused_distance_m = 0
    
    # Step-based distance
    step_increment = 0  # Default in caso non venga calcolato
    if step_count is not None and len(step_history) > 0:
        last_timestamp, last_step_count, _ = step_history[-1]  # Fixed: Correctly unpack the tuple
        step_increment = max(0, step_count - last_step_count)  # Ensure non-negative
        step_distance_m = steps_to_distance(step_increment, user_height, user_age, user_gender)
    
    # Position-based distance calculation
    if estimated_x is not None and estimated_y is not None and len(position_history) > 0:
        last_timestamp, last_x, last_y, _, _ = position_history[-1]  # Fixed: Correctly unpack the tuple
        position_distance_m = calculate_distance_between_points(last_x, last_y, estimated_x, estimated_y)
        
        # Apply simple noise filtering - ignore unrealistic movements
        # (e.g., if position jumps more than 2 meters in a second)
        time_diff = (current_timestamp - last_timestamp).total_seconds()
        max_reasonable_speed = 2.0  # meters per second (normal walking)
        max_reasonable_distance = max_reasonable_speed * time_diff
        
        if position_distance_m > max_reasonable_distance:
            # Likely a measurement error, so use previous position
            position_distance_m = 0
            estimated_x, estimated_y = last_x, last_y
    
    # If wearable seems stationary, block small movements
    MIN_MOVEMENT_THRESHOLD = 0.2  # meters = 20cm

    if len(position_history) > 0:
        last_timestamp, last_x, last_y, _, _ = position_history[-1]
        if position_distance_m < MIN_MOVEMENT_THRESHOLD and step_increment == 0:
            estimated_x, estimated_y = last_x, last_y
            position_distance_m = 0

    # Fuse distance measurements
    if step_distance_m > 0 or position_distance_m > 0:
        # Dynamic confidence based on number of anchors with valid readings
        valid_anchor_count = sum(1 for d in distances.values() if d > 0)

        # Calcola il peso dinamico per la posizione, ma limita al massimo 40%
        max_position_weight = 0.3

        confidence_position = min(max_position_weight, valid_anchor_count / len(ANCHOR_POSITIONS) * max_position_weight)
        
        # Assicura che lo step abbia sempre almeno il 70% di peso
        confidence_step = 1 - confidence_position
        
        fused_distance_m = fuse_distance_measurements(
            step_distance_m, 
            position_distance_m,
            confidence_step=confidence_step,
            confidence_position=confidence_position
        )
    
    # Initialize history if it's empty
    if not step_history and step_count is not None:
        step_history.append((current_timestamp, 0, 0))  # Start with 0 steps and 0 distance
    
    if not position_history and estimated_x is not None and estimated_y is not None:
        position_history.append((current_timestamp, estimated_x, estimated_y, 0, 0))  # Start with 0 distance
    
    # Calculate total distances
    total_step_distance_m = sum(dist for _, _, dist in step_history) + step_distance_m
    total_position_distance_m = sum(dist for _, _, _, dist, _ in position_history) + position_distance_m
    total_fused_distance_m = sum(dist for _, _, _, _, dist in position_history) + fused_distance_m
    
    # Update history with new measurements
    if step_count is not None:
        step_history.append((current_timestamp, step_count, step_distance_m))
    
    if estimated_x is not None and estimated_y is not None:
        position_history.append((current_timestamp, estimated_x, estimated_y, position_distance_m, fused_distance_m))
    
    # Prepare row data
    row_data = {
        'timestamp': current_timestamp.isoformat(),
        'estimated_x': estimated_x if estimated_x is not None else "",
        'estimated_y': estimated_y if estimated_y is not None else "",
        'steps': step_count if step_count is not None else "",
        'step_distance_m': round(step_distance_m, 3),
        'position_distance_m': round(position_distance_m, 3),
        'fused_distance_m': round(fused_distance_m, 3),
        'total_step_distance_m': round(total_step_distance_m, 3),
        'total_position_distance_m': round(total_position_distance_m, 3),
        'total_fused_distance_m': round(total_fused_distance_m, 3)
    }
    
    # Add data from each anchor
    for anchor_id in ANCHOR_POSITIONS.keys():
        if anchor_id in latest_data:
            data = latest_data[anchor_id]
            row_data[f'distance_{anchor_id}'] = data.get('distance', "")
            row_data[f'rssi_{anchor_id}'] = data.get('rssi', "")
        else:
            row_data[f'distance_{anchor_id}'] = ""
            row_data[f'rssi_{anchor_id}'] = ""
    
    # Write to CSV
    try:
        with open(csv_path, 'a', newline='') as f:
            # Get list of column names from the CSV file
            with open(csv_path, 'r') as read_f:
                reader = csv.reader(read_f)
                headers = next(reader)  # Read the header row
            
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row_data)
            print(f"CSV updated at {current_timestamp.isoformat()}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
    
    # Print status update
    if estimated_x is not None and step_count is not None:
        print(f"Position: ({estimated_x}, {estimated_y}), Steps: {step_count}")
        print(f"Distance (step): {step_distance_m:.2f}m, (position): {position_distance_m:.2f}m, (fused): {fused_distance_m:.2f}m")
        print(f"Total distance: {total_fused_distance_m:.2f}m ({total_fused_distance_m/1000:.3f}km)")

# Handle client connection from an anchor
def handle_client(client_socket, latest_data, csv_update_event):
    try:
        # Receive data
        data_raw = client_socket.recv(1024)
        if not data_raw:
            return
        
        # Decode JSON
        data = json.loads(data_raw.decode())
        
        print("new pkt received at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])

        # Extract information
        anchor_id = data.get('anchor_id')
        if anchor_id:
            # Update the dictionary of latest data
            latest_data[anchor_id] = {
                'timestamp': data.get('timestamp'),
                'rssi': data.get('rssi'),
                'distance': data.get('distance'),
                'steps': data.get('steps')
            }
            
            print(f"Received data from {anchor_id}: RSSI={data.get('rssi')}, "
                  f"Distance={data.get('distance')}m, Steps={data.get('steps')}")
            
            # Signal that we have new data to process
            csv_update_event.set()
            
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()

# Start server
def start_server(patient_data):

    PATIENT_ID = patient_data['id']
    USER_HEIGHT = patient_data['height']
    USER_GENDER = patient_data['gender']
    USER_AGE = patient_data['age']

    # Create directory and CSV file
    folder_name = create_data_directory(PATIENT_ID, patient_data)
    csv_path = initialize_csv(folder_name)
    
    # Dictionary to maintain latest data from each anchor
    latest_data = {}
    
    # Event to signal when new data is available
    csv_update_event = threading.Event()
    
    # History for distance calculation - corrected tuples structure
    position_history = []  # Format: [(timestamp, x, y, distance, fused_distance), ...]
    step_history = []      # Format: [(timestamp, step_count, step_distance), ...]
    
    # Initialize server socket
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:

        # Kalman filters for position smoothing
        kalman_x = Kalman2D()
        kalman_y = Kalman2D()

        server.bind((HOST, PORT))
        server.listen(5)
        print(f"Server listening on {HOST}:{PORT}")
        print(f"User settings: ID={PATIENT_ID}, Height={USER_HEIGHT}m, Age={USER_AGE}, Gender={USER_GENDER}")
        
        # Separate thread to update CSV
        def update_csv_thread():
            last_update_time = time.time()
            
            while True:
                # Wait for signal that new data is available or timeout after 1 second
                csv_update_event.wait(timeout=0.25)
                
                current_time = time.time()
                # Update CSV either when we have new data or at least every second
                if csv_update_event.is_set() or (current_time - last_update_time) >= 1.0:
                    update_csv(csv_path, latest_data, position_history, step_history, kalman_x, kalman_y, USER_HEIGHT, USER_AGE, USER_GENDER)
                    last_update_time = current_time
                    csv_update_event.clear()
        
        # Start CSV update thread
        csv_thread = threading.Thread(target=update_csv_thread)
        csv_thread.daemon = True
        csv_thread.start()
        
        # Main loop to accept connections
        while True:
            client, addr = server.accept()
            print(f"Connection accepted from {addr[0]}:{addr[1]}")
            
            # Handle client in separate thread
            client_thread = threading.Thread(
                target=handle_client, 
                args=(client, latest_data, csv_update_event)
            )
            client_thread.daemon = True
            client_thread.start()
            
    except KeyboardInterrupt:
        print("Server terminated by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()

if __name__ == "__main__":

    # Aspetta che il file venga creato dal frontend
    while not os.path.exists("current_patient.json"):
        print("Waiting for current_patient.json...")
        time.sleep(1)

    with open("current_patient.json", "r") as f:
        patient_data = json.load(f)

    # Dopo il test, elimina il file per evitare riusi
    try:
        os.remove("current_patient.json")
        print("Removed current_patient.json after test.")
    except Exception as e:
        print(f"Warning: could not remove patient JSON: {e}")

    print("Starting enhanced distance tracking server...")
    start_server(patient_data)


