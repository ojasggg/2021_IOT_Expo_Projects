import serial
import time
import string
import pynmea2
import smtplib
from math import radians, sin, cos, acos
import os

while True:
    port=("/dev/ttyAMA0")
    ser=serial.Serial(port, baudrate=9600, timeout=0.5)
        #dataout = pynmea2.NMEAStreamReader()
    newdata=ser.readline()
   

    if newdata[0:6] == "$GPRMC":
        newmsg=pynmea2.parse(newdata)
        lat=newmsg.latitude
        lng=newmsg.longitude
        os.system("python3 TFLite_detection_webcam.py")
        
        
        if (lat or lng) == 0:
            continue
        
        else:
            slat = radians(27.7040449)
            slon = radians(85.3273726)
            elat = radians(lat)
            elon = radians(lng)

            dist = 6371.01 * acos(sin(slat)*sin(elat) + cos(slat)*cos(elat)*cos(slon - elon))
            location = "Latitude =" + str(lat) + " and Longlitude =" + str(lng)
            print(dist)
            
            if dist > 10:
                
                sender_email= "brightnight26a@gmail.com"
                rec_email=["joshikamala2121@gmail.com","bishalbidari789@gmail.com","david.joshi013@gmail.com","liveprasthan08@gmail.com","ojasggg@gmail.com"]
                password = ("smartglasses12345")
    #             message= "Your relative is in " + location + " Location. /n Please Search his/her location in https://www.google.com/maps"
            
                message = "Testing"
                
                server=smtplib.SMTP('smtp.gmail.com', 587)
                
                server.starttls()
                server.login(sender_email, password)
                print("loged in")
                server.sendmail(sender_email,rec_email, message)
                print("email sent to", rec_email)
                break
            else:
                pass
   
            
                    
            
    
        

