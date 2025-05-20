#!/usr/bin/env python3
import socket
import json
import time
import math
import random
import threading
import argparse
from datetime import datetime

# Configuration
DEFAULT_SERVER_IP = "127.0.0.1"  # Local server for testing
DEFAULT_SERVER_PORT = 5000

# Anchor positions (same as in the real system)
ANCHOR_POSITIONS = {
    'anchor1': (0, 0),
    'anchor2': (10, 0),
    'anchor3': (10, 10),
    'anchor4': (0, 10)
}

# RSSI parameters for simulation
RSSI_AT_1M = -60
PATH_LOSS_EXPONENT = 3.0
RSSI_NOISE_STD_DEV = 3.0  # Standard deviation for RSSI noise (dBm)

# Wearable parameters for simulation
STEP_LENGTH = 0.75  # meters
STEPS_PER_SECOND_WALKING = 2.0  # average steps per second when walking
PROBABILITY_OF_STANDING_STILL = 0.2  # probability of standing still in each update

# Calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Calculate RSSI from distance with realistic noise
def calculate_rssi_from_distance(distance):
    # Use log-distance path loss model: RSSI = RSSI_at_1m - 10 * N * log10(d)
    if distance <= 0:
        return RSSI_AT_1M
    
    # Calculate ideal RSSI
    ideal_rssi = RSSI_AT_1M - 10 * PATH_LOSS_EXPONENT * math.log10(distance)
    
    # Add realistic noise
    noise = random.gauss(0, RSSI_NOISE_STD_DEV)
    
    return round(ideal_rssi + noise)

# Simulate wearable movement
class WearableSimulator:
    def __init__(self, room_size=(10, 10), update_interval=1.0):
        self.room_size = room_size
        self.update_interval = update_interval
        
        # Starting position (random within the room)
        self.x = random.uniform(0, room_size[0])
        self.y = random.uniform(0, room_size[1])
        
        # Movement parameters
        self.speed = 0.5  # meters per second
        self.direction = random.uniform(0, 2 * math.pi)  # random direction
        self.step_count = 0
        self.standing_still = False
        
        # Start the simulation thread
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _simulation_loop(self):
        last_update = time.time()
        
        while self.running:
            current_time = time.time()
            time_delta = current_time - last_update
            
            # Randomly decide if standing still or walking
            if random.random() < PROBABILITY_OF_STANDING_STILL:
                self.standing_still = True
            else:
                self.standing_still = False
                
                # Randomly change direction occasionally
                if random.random() < 0.1:
                    self.direction = random.uniform(0, 2 * math.pi)
                
                # Move in the current direction
                distance = self.speed * time_delta
                self.x += distance * math.cos(self.direction)
                self.y += distance * math.sin(self.direction)
                
                # Add steps based on distance
                new_steps = int(distance / STEP_LENGTH)
                self.step_count += new_steps
                
                # Bounce off walls
                if self.x < 0:
                    self.x = -self.x
                    self.direction = math.pi - self.direction
                elif self.x > self.room_size[0]:
                    self.x = 2 * self.room_size[0] - self.x
                    self.direction = math.pi - self.direction
                
                if self.y < 0:
                    self.y = -self.y
                    self.direction = 2 * math.pi - self.direction
                elif self.y > self.room_size[1]:
                    self.y = 2 * self.room_size[1] - self.y
                    self.direction = 2 * math.pi - self.direction
            
            last_update = current_time
            time.sleep(0.1)  # Update position 10 times per second
    
    def get_position(self):
        return (self.x, self.y)
    
    def get_step_count(self):
        return self.step_count
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

# Simulate an anchor node
class AnchorSimulator:
    def __init__(self, anchor_id, position, server_ip, server_port, wearable, update_interval=1.0):
        self.anchor_id = anchor_id
        self.position = position
        self.server_ip = server_ip
        self.server_port = server_port
        self.wearable = wearable
        self.update_interval = update_interval
        
        # Start the simulation thread
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _simulation_loop(self):
        while self.running:
            try:
                # Get current wearable position and step count
                wearable_pos = self.wearable.get_position()
                step_count = self.wearable.get_step_count()
                
                # Calculate true distance to wearable
                distance = calculate_distance(
                    self.position[0], self.position[1],
                    wearable_pos[0], wearable_pos[1]
                )
                
                # Calculate RSSI with realistic noise
                rssi = calculate_rssi_from_distance(distance)
                
                # Prepare data packet
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'anchor_id': self.anchor_id,
                    'anchor_position': self.position,
                    'wearable_id': 'AA:BB:CC:DD:EE:FF',  # Simulated MAC address
                    'rssi': rssi,
                    'distance': round(distance, 2),
                    'steps': step_count
                }
                
                # Send data to server
                self._send_data_to_server(data)
                
                print(f"Anchor {self.anchor_id} - Distance: {distance:.2f}m, RSSI: {rssi} dBm, Steps: {step_count}")
            
            except Exception as e:
                print(f"Error in anchor {self.anchor_id}: {e}")
            
            time.sleep(self.update_interval)
    
    def _send_data_to_server(self, data):
        try:
            # Create a socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.server_ip, self.server_port))
            
            # Send JSON data
            sock.sendall(json.dumps(data).encode())
            
            # Close the socket
            sock.close()
        except Exception as e:
            print(f"Failed to send data to server: {e}")
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)

def main():
    parser = argparse.ArgumentParser(description='BLE Positioning System Simulator')
    parser.add_argument('--server', default=DEFAULT_SERVER_IP, help='Server IP address')
    parser.add_argument('--port', type=int, default=DEFAULT_SERVER_PORT, help='Server port')
    parser.add_argument('--interval', type=float, default=1.0, help='Update interval in seconds')
    args = parser.parse_args()
    
    try:
        print(f"Starting BLE positioning system simulator...")
        print(f"Sending data to server at {args.server}:{args.port}")
        print(f"Update interval: {args.interval} seconds")
        print("Press Ctrl+C to stop the simulation")
        
        # Create wearable simulator
        wearable = WearableSimulator(room_size=(10, 10), update_interval=args.interval)
        
        # Create anchor simulators
        anchors = []
        for anchor_id, position in ANCHOR_POSITIONS.items():
            anchor = AnchorSimulator(
                anchor_id=anchor_id,
                position=position, 
                server_ip=args.server,
                server_port=args.port,
                wearable=wearable,
                update_interval=args.interval
            )
            anchors.append(anchor)
        
        # Keep the main thread running
        while True:
            wearable_pos = wearable.get_position()
            step_count = wearable.get_step_count()
            print(f"\nWearable Position: ({wearable_pos[0]:.2f}, {wearable_pos[1]:.2f})")
            print(f"Step Count: {step_count}")
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        # Stop all simulators
        if 'wearable' in locals():
            wearable.stop()
        
        if 'anchors' in locals():
            for anchor in anchors:
                anchor.stop()
        
        print("Simulation stopped.")

if __name__ == "__main__":
    main()
