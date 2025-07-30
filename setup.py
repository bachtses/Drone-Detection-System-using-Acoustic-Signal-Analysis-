import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=2)

def send_cmd(cmd):
    ser.write((cmd + '\r\n').encode())
    time.sleep(0.2)
    print(ser.read_all().decode(errors='ignore'))

send_cmd('+++')
send_cmd('AT+RESTORE=1')  # factory reset
time.sleep(1)

# Enter AT mode
send_cmd('+++')
time.sleep(1)

# Set parameters
send_cmd('AT+MODE=1')        # Stream mode
send_cmd('AT+SF=7')
send_cmd('AT+BW=0')          # 125kHz
send_cmd('AT+CR=1')          # 4/5 coding rate
send_cmd('AT+PWR=22')        # Max power
send_cmd('AT+NETID=0')
send_cmd('AT+TXCH=18')
send_cmd('AT+RXCH=18')
send_cmd('AT+BAUD=115200')
send_cmd('AT+COMM="8N1"')
send_cmd('AT+RSSI=1')        # Optional
send_cmd('AT+EXIT')          # Exit AT mode

ser.close()

