import RPi.GPIO as GPIO
import time
import os
import random

GPIO.setmode(GPIO.BCM) #use the GPIO numbering
GPIO.setwarnings(False) # Avoids warning channel is already in use

led = 21 # GPIO pin 21
button = 18 # GPIO pin 18
button_led = 26 # GPIO pin 26

GPIO.setup(led,GPIO.OUT) # sets up pin 21 to output
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_UP) #sets up pin 18 as a button
GPIO.setup(button_led,GPIO.OUT) #sets up pin 26 to output



i_count = 0 # set up for correct grammar in notification below

while True:
        GPIO.output(button_led, True) # turn on button led
        input_state = GPIO.input(button) # primes the button!
        if input_state == False:
            i_count = i_count + 1
            GPIO.output(button_led, False) # turns off button led
            GPIO.output(led,True) #Turn on LED
          
            GPIO.output(led,False) #turn off LED
            if i_count == 1:
                print("History Bot has been activated " + str(i_count) + " time!")
            else:
                print("History Bot has been activated " + str(i_count) + " times!")
            GPIO.output(button_led, True) # turns button led back on
            time.sleep(0.2)