# SIMSMS1.py

import RPi.GPIO as GPIO
import serial
import time, sys
import datetime

P_BUTTON = 24 # Button, adapt to your wiring

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(P_BUTTON, GPIO.IN, GPIO.PUD_UP)

SERIAL_PORT = "/dev/ttyAMA0"  # Raspberry Pi 2
#SERIAL_PORT = "/dev/ttyS0"    # Raspberry Pi 3

ser = serial.Serial(SERIAL_PORT, baudrate = 9600, timeout = 5)
setup()
ser.write("AT+CMGF=1\r") # set to text mode
time.sleep(3)
ser.write('AT+CMGDA="DEL ALL"\r') # delete all SMS
time.sleep(3)
reply = ser.read(ser.inWaiting()) # Clean buf
print "Listening for incomming SMS..."
while True:
    reply = ser.read(ser.inWaiting())
    if reply != "":
        ser.write("AT+CMGR=1\r") 
        time.sleep(3)
        reply = ser.read(ser.inWaiting())
        print "SMS received. Content:"
        print reply
        if "getStatus" in reply:
            t = str(datetime.datetime.now())
            if GPIO.input(P_BUTTON) == GPIO.HIGH:
                state = "Button released"
            else:
                state = "Button pressed"
            ser.write('AT+CMGS="+41764331356"\r')
            time.sleep(3)
            msg = "Sending status at " + t + ":--" + state
            print "Sending SMS with status info:" + msg
            ser.write(msg + chr(26))
        time.sleep(3)
        ser.write('AT+CMGDA="DEL ALL"\r') # delete all
        time.sleep(3)
        ser.read(ser.inWaiting()) # Clear buf
    time.sleep(5)    

