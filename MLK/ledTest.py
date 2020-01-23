import time
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(40, GPIO.OUT)
while True:
    GPIO.output(40,True)
    time.sleep(1)
    GPIO.output(40,False)
    time.sleep(1)