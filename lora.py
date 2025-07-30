import serial
import time


class LoRaModule:
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"\033[96m[LoRa] Serial port {port} opened.\033[0m")
        except Exception as e:
            print(f"\033[91m[LoRa INIT ERROR] Could not open serial port: {e}\033[0m")
            self.ser = None

    def send_command(self, command, delay=0.2):
        if self.ser is None:
            return ""
        if not self.ser.is_open:
            self.ser.open()
        self.ser.write((command + '\r\n').encode())
        time.sleep(delay)
        return self.ser.read_all().decode(errors='ignore').strip()

    def enter_at_mode(self):
        if self.ser is None:
            return ""
        self.ser.write(b'+++')
        time.sleep(1)
        return self.ser.read_all().decode(errors='ignore').strip()

    def exit_at_mode(self):
        return self.send_command('AT+EXIT')

    def setup_module(self):
        print("\033[96m[LoRa] Configuring module...\033[0m")
        self.enter_at_mode()
        self.send_command('AT+MODE=1')         # Stream mode
        self.send_command('AT+SF=7')           # Spreading factor
        self.send_command('AT+BW=0')           # Bandwidth: 125 kHz
        self.send_command('AT+CR=1')           # Coding rate: 4/5
        self.send_command('AT+PWR=22')         # Max power
        self.send_command('AT+NETID=0')
        self.send_command('AT+TXCH=18')
        self.send_command('AT+RXCH=18')
        self.send_command('AT+BAUD=115200')
        self.send_command('AT+COMM="8N1"')
        self.send_command('AT+RSSI=0')         # Disable RSSI output
        self.exit_at_mode()
        print("\033[92m[LoRa] Module configured successfully.\033[0m")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def is_open(self):
        return self.ser and self.ser.is_open

    def flush(self):
        if self.ser:
            self.ser.flushInput()
            self.ser.flushOutput()


class LoRaTransmitter(LoRaModule):
    def send_message(self, message):
        if self.ser is None:
            print(f"\033[91m[LoRa TX ERROR] Serial port not initialized.\033[0m")
            return
        try:
            #print(f"\033[94m[LoRa TX] Attempting to send: {message}\033[0m")
            self.ser.write((message + '\n').encode())
            self.ser.flush()
            #print(f"\033[92m[LoRa TX] Message sent successfully.\033[0m")
        except Exception as e:
            print(f"\033[91m[LoRa TX ERROR] {e}\033[0m")


class LoRaReceiver(LoRaModule):
    def listen(self):
        if self.ser is None:
            print("\033[91m[LoRa RX ERROR] Serial port not initialized.\033[0m")
            return
        print("\033[96m[LoRa RX] Listening for messages...\033[0m")
        try:
            while True:
                if self.ser.in_waiting:
                    data = self.ser.readline().decode(errors='ignore').strip()
                    if data and not (data.isdigit() and int(data) in range(30, 120)):
                        print(f"\033[92m[LoRa RX] Received: {data}\033[0m")
                        yield data
        except KeyboardInterrupt:
            print("\n\033[96m[LoRa RX] Stopped.\033[0m")
        finally:
            self.close()
