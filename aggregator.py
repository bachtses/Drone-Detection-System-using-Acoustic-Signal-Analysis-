import socket
import threading
import time
import json
from math import atan2, degrees, radians, cos, sin
import folium
import datetime
from lora import LoRaReceiver

####################################################################################
#################    CONFIGURATION  (Ports, Nodes, Coordinates)    #################
####################################################################################
HOST = 'X.X.X.X'  # To accept connections from all interfaces
AGGREGATOR_PORT = XXXX

CENTRAL_NODE_IP = 'XXX.XX.XX.XX'
CENTRAL_NODE_PORT = XXXX

NODES = {
    "2": {"name": "Node 2", "location": (22.998053, 40.568898)},
    "3": {"name": "Node 3", "location": (22.998523, 40.568539)}
} 

MAIN_AGGREGATOR_LOCATION = (22.998053, 40.568539)

last_detections = {
    node_info["name"]: {"prob": 0.0, "timestamp": 0, "spectrogram": None}
    for node_info in NODES.values()
}

central_conn = None
central_conn_lock = threading.Lock()

def connect_to_central_node():
    global central_conn
    connection_attempted = False   

    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((CENTRAL_NODE_IP, CENTRAL_NODE_PORT))
            with central_conn_lock:
                central_conn = s
            print("\033[96m[Aggregator] Connected to Central Node \033[0m")
            return
        except Exception:
            if not connection_attempted:
                print("\033[91m[Aggregator] Central Node down. Retrying... \033[0m")
                connection_attempted = True
            time.sleep(5)


####################################################################################
####################             HTML MAP GENERATION             ###################
####################################################################################
def generate_node_map(nodes, central_location, output_file="topology_map.html"):
    map_center = central_location[::-1]
    node_map = folium.Map(location=map_center, zoom_start=19, tiles='CartoDB positron')

    folium.Marker(
        location=(central_location[1], central_location[0]),
        popup="Aggregator",
        icon=folium.Icon(color='blue', icon='home', prefix='fa')
    ).add_to(node_map)

    folium.map.Marker(
        location=(central_location[1], central_location[0]),
        icon=folium.DivIcon(
            html=f"""<div style="font-size: 12px; color: blue; font-weight: bold; margin-left: -25px; margin-top: 10px;">Aggregator</div>"""
        )
    ).add_to(node_map)

    for port, info in nodes.items():
        lon, lat = info["location"]
        name = info["name"]

        icon = folium.Icon(color='darkblue', icon='circle', prefix='fa')

        folium.Marker(
            location=(lat, lon),
            popup=f"{name} (Port {port})",
            icon=icon
        ).add_to(node_map)

        folium.map.Marker(
            location=(lat, lon),
            icon=folium.DivIcon(
                html=f"""<div style="width: 100px; font-size: 12px; color: blue; font-weight: bold; margin-left: -15px; margin-top: 10px;">{name}</div>"""
            )
        ).add_to(node_map)

    node_map.save(output_file)
    print(f"\033[96m[Map] Node map exported to {output_file}\033[0m")



####################################################################################
####################              DIRECTION OF SOUND            ####################
####################################################################################
def calculate_direction(node_name):
    for node_info in NODES.values():
        if node_info["name"] == node_name:
            lon2, lat2 = node_info["location"]
            break
    else:
        return "Unknown"
    lon1, lat1 = MAIN_AGGREGATOR_LOCATION
    d_lon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    x = sin(d_lon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1) * cos(lat2) * cos(d_lon))
    bearing = degrees(atan2(x, y))
    bearing = (bearing + 360) % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    idx = round(bearing / 45)
    return directions[idx]

####################################################################################
##############     CENTRAL NODE CONNECTION AND PERIPHERAL FUSION     ###############
####################################################################################
def central_node_connection_and_peripheral_fusion():
    global central_conn  
    threading.Thread(target=connect_to_central_node, daemon=True).start()
    while True:
        time.sleep(1)
        current_time = time.time()
        valid_detections = {
            node: info for node, info in last_detections.items()
            if current_time - info["timestamp"] < 2
        }

        if valid_detections:
            best_node = max(valid_detections.items(), key=lambda x: x[1]["prob"])
            node_name = best_node[0]
            prob = best_node[1]["prob"]
            direction = calculate_direction(node_name)
            angle = best_node[1].get("angle", -1)
            timestamp = datetime.datetime.fromtimestamp(best_node[1]["timestamp"]).strftime('%H:%M:%S')

            print(f"[Aggregator] Drone Detected | {node_name} | {prob:.2f} | {direction} | {angle}° | {timestamp}")

            with central_conn_lock:
                if central_conn:
                    try:
                        message = json.dumps({
                            "node": node_name,
                            "probability": prob,
                            "direction": direction,
                            "angle": angle,
                            "timestamp": timestamp
                        }) + '\n'
                        central_conn.sendall(message.encode())
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("\033[91m[Aggregator] Central Node down. Retrying... \033[0m")
                        try:
                            central_conn.close()
                        except:
                            pass
                        central_conn = None
                        threading.Thread(target=connect_to_central_node, daemon=True).start()



####################################################################################
####################              AGGREGATOR SERVER             ####################
####################################################################################
def start_aggregator_server():
    
    print("\033[96m[Aggregator] Ready to receive LoRa messages...\033[0m")
    lora = LoRaReceiver()
    lora.setup_module() 

    for raw_data in lora.listen():
        try:
            data = raw_data.decode('utf-8') if isinstance(raw_data, bytes) else raw_data

            # Skip short or obviously invalid messages
            if not data.strip().startswith('{'):
                continue

            msg = json.loads(data)

            node_id = str(msg.get("node_id", "unknown"))
            label = msg.get("label")
            prob = msg.get("probability")
            angle = msg.get("angle", -1)
            spectrogram = msg.get("spectrogram", None)
            node_info = NODES.get(node_id.lower(), {"name": f"Node {node_id}", "location": (0, 0)})
            node_name = node_info["name"]

            if label == "Drone":
                last_detections[node_name] = {
                    "prob": prob,
                    "timestamp": time.time(),
                    "spectrogram": spectrogram,
                    "angle": angle
                }
            
            #print(f"\033[96m[Aggregator] [LoRa] Message received from {node_name}: {label}, {prob}\033[0m")

        except Exception as e:
            print(f"\033[91m[Aggregator] Failed to process LoRa message: {e}\033[0m")

####################################################################################
####################                     MAIN                   ####################
####################################################################################
if __name__ == "__main__":
    #generate_node_map(NODES, MAIN_AGGREGATOR_LOCATION)
    threading.Thread(target=central_node_connection_and_peripheral_fusion, daemon=True).start()

    try:
        start_aggregator_server()
    except KeyboardInterrupt:
        print("\n\033[96m[Aggregator] Shutting down...\033[0m")
        exit(0)
