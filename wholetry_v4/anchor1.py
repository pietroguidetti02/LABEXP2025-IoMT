#!/usr/bin/env python3
import asyncio
import json
import socket
import time
import argparse
import struct
from bleak import BleakScanner

# Configuration
SERVER_HOST = '192.168.116.69'  # Your PC's IP
SERVER_PORT = 5000
ANCHOR_ID = 'anchor1'  # Change for each anchor
SCAN_INTERVAL = 0.1  # seconds
MANUFACTURER_ID = 0xFFFF  # Match the ID used in the wearable

# Parse command line arguments
parser = argparse.ArgumentParser(description='BLE Scanner and Forwarder')
parser.add_argument('--id', type=str, help='Anchor ID (anchor1, anchor2, etc.)')
parser.add_argument('--server', type=str, help='Server IP address')
args = parser.parse_args()

if args.id:
    ANCHOR_ID = args.id
if args.server:
    SERVER_HOST = args.server

print(f"Starting anchor {ANCHOR_ID}, sending data to {SERVER_HOST}:{SERVER_PORT}")

'''
async def scan_and_forward():
    while True:
        try:
            print("Scanning for devices...")
            # Scan for BLE devices
            devices = await BleakScanner.discover(SCAN_INTERVAL, return_adv=True)

            print(f"Found {len(devices)} devices")
            # Look for our wearable device
            for d, (device, adv_data) in devices.items():
                # Print all device names to help with debugging
                if device.name:
                    print(f"Found device: {device.name}")

                # Our wearable should be named "StepTracker"
                if device.name == "StepTracker":
                    print(f"Found StepTracker device: {device.address}")

                    # Extract RSSI
                    rssi = adv_data.rssi
                    print(f"RSSI: {rssi}")

                    # Debug the advertisement data
                    print(f"Advertisement data: {adv_data.manufacturer_data}")

                    # Extract manufacturer data if available
                    steps = None

                    # Look for our specific manufacturer ID in the data
                    if MANUFACTURER_ID in adv_data.manufacturer_data:
                        data_bytes = adv_data.manufacturer_data[MANUFACTURER_ID]
                        print(f"Manufacturer data: {data_bytes.hex()}")

                        # Check if data format matches what we expect
                        if len(data_bytes) >= 5 and data_bytes[0] == 0x01:
                            # Extract steps (4 bytes)
                            steps = int.from_bytes(data_bytes[1:5], byteorder='little')
                            print(f"Parsed steps: {steps}")

                    # Only send data if we have valid steps data
                    if steps is not None:
                        # Calculate approximate distance based on RSSI
                        tx_power = -59  # Typical value, should be calibrated
                        n = 2.0  # Path loss exponent
                        distance = 10 ** ((tx_power - rssi) / (10 * n))

                        # Prepare data to send to server
                        data = {
                            'anchor_id': ANCHOR_ID,
                            'timestamp': time.time(),
                            'rssi': rssi,
                            'distance': round(distance, 2),
                            'steps': steps
                        }

                        # Send data to server
                        await send_to_server(data)
                        print(f"Sent data: {data}")
                    else:
                        print("No step data found in advertisement")

                        # Try to connect and read characteristic as fallback
                        try:
                            print("Attempting to connect and read step characteristic...")
                            steps = await read_step_characteristic(device.address)

                            if steps is not None:
                                # Calculate distance based on RSSI
                                tx_power = -59
                                n = 2.0
                                distance = 10 ** ((tx_power - rssi) / (10 * n))

                                # Prepare data to send to server
                                data = {
                                    'anchor_id': ANCHOR_ID,
                                    'timestamp': time.time(),
                                    'rssi': rssi,
                                    'distance': round(distance, 2),
                                    'steps': steps
                                }

                                # Send data to server
                                await send_to_server(data)
                                print(f"Sent data (from characteristic): {data}")
                        except Exception as e:
                            print(f"Error reading characteristic: {e}")

        except Exception as e:
            print(f"Error in scan_and_forward: {e}")

        await asyncio.sleep(1)
'''

async def scan_and_forward():
    while True:
        try:
            # Scan for BLE devices
            devices = await BleakScanner.discover(SCAN_INTERVAL, return_adv=True)

            for _, (device, adv_data) in devices.items():
                if device.name != "StepTracker":
                    continue  # Ignora tutti tranne StepTracker

                rssi = adv_data.rssi
                steps = None

                if MANUFACTURER_ID in adv_data.manufacturer_data:
                    data_bytes = adv_data.manufacturer_data[MANUFACTURER_ID]

                    if len(data_bytes) >= 5 and data_bytes[0] == 0x01:
                        steps = int.from_bytes(data_bytes[1:5], byteorder='little')

                if steps is not None:
                    # Calcola distanza approssimativa
                    tx_power = -59
                    n = 2.0
                    distance = 10 ** ((tx_power - rssi) / (10 * n))

                    data = {
                        'anchor_id': ANCHOR_ID,
                        'timestamp': time.time(),
                        'rssi': rssi,
                        'distance': round(distance, 2),
                        'steps': steps
                    }

                    #print data ricevuti
                    print("Data received:")
                    print(json.dumps(data, indent=4))

                    await send_to_server(data)
                else:
                    # Fall back: prova a leggere caratteristica se non c'è advertising
                    try:
                        steps = await read_step_characteristic(device.address)
                        if steps is not None:
                            distance = 10 ** ((-59 - rssi) / (10 * 2.0))
                            timestamp_temp = time.time()
                            data = {
                                'anchor_id': ANCHOR_ID,
                                'timestamp': timestamp_temp,
                                'rssi': rssi,
                                'distance': round(distance, 2),
                                'steps': steps
                            }

                            #print data ricevuti
                            print("Data received:")
                            print(json.dumps(data, indent=4))

                            await send_to_server(data)
                    except Exception:
                        pass

        except Exception as e:
            print(f"Error in scan_and_forward: {e}")

        await asyncio.sleep(0.5)


async def read_step_characteristic(device_address):
    from bleak import BleakClient

    # Step service and characteristic UUIDs (must match wearable)
    STEP_SERVICE_UUID = "00001234-0000-1000-8000-00805f9b34fb"
    STEP_CHAR_UUID = "00002345-0000-1000-8000-00805f9b34fb"

    try:
        async with BleakClient(device_address, timeout=5.0) as client:
            print("Connected to device")

            # Read the step characteristic
            value = await client.read_gatt_char(STEP_CHAR_UUID)

            # Convert bytes to steps count
            steps = int.from_bytes(value, byteorder='little')
            print(f"Read steps from characteristic: {steps}")
            #print(f"RSSI: {rssi} | distance: {distance}")

            return steps
    except Exception as e:
        print(f"Failed to read characteristic: {e}")
        return None

async def send_to_server(data):
    try:
        # Create TCP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)  # Add timeout to avoid hanging
        sock.connect((SERVER_HOST, SERVER_PORT))

        # Convert to JSON and send
        json_data = json.dumps(data)
        sock.sendall(json_data.encode())

        # Close socket
        sock.close()
        print("Data sent successfully: ")
        print(json.dumps(data, indent=4))  # <-- stampa leggibile

    except Exception as e:
        print(f"Error sending to server: {e}")

# Main entry point
if __name__ == "__main__":
    try:
        asyncio.run(scan_and_forward())
    except KeyboardInterrupt:
        print("Program terminated by user")