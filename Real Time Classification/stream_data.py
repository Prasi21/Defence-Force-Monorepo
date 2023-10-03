
# Requires installation of the correct Movella DOT PC SDK wheel through pip
# For example, for Python 3.9 on Windows 64 bit run the following command
# pip install movelladot_pc_sdk-202x.x.x-cp39-none-win_amd64.whl

# This code is based off movelladot_pc_sdk_receive_data.py example provdied by Movella

import threading
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import Label
import queue
from xdpchandler import *
import keyboard as keyboard_scanner
import keras

endStream = False
reset_headers = True

# MODEL_NAME = "model_M4_all.keras"
MODEL_NAME = "model_label_M4.h5"

all_classes = ['None', 'aim', 'cocking', 'extreme low ready', 'high ready',
    'low ready', 'magazine change', 'magazine recovery', 'scanning', 'shot']
 
mlb = MultiLabelBinarizer(classes = all_classes)
mlb.fit(all_classes)

color_dict = {
    'None': '#FFFFFF',              # Black
    'aim': '#FF5733',               # Orange
    'cocking': '#FFC300',           # Yellow
    'extreme low ready': '#3498DB', # Blue
    'high ready': '#27AE60',        # Green
    'low ready': '#8E44AD',         # Purple
    'magazine change': '#E74C3C',   # Red
    'magazine recovery': '#16A085', # Dark Green
    'scanning': '#2ECC71',          # Emerald Green
    'shot': '#D35400',              # Pumpkin
    'Initialising': '#6C7A89'       # Grayish Blue
}


# Queue to send info from thread to gui
blockingQueue = queue.Queue(maxsize=1) # Maxsize to prevent buildup of old actions


def on_press(key):
    if key == keyboard.Key.esc:
        global endStream
        endStream = True
    elif key == keyboard.Key.space:
        global reset_headers
        reset_headers = True

# Data collection is run on a separate thread
def collect_data():
    # IMU initialisation Code
    xdpcHandler = XdpcHandler()

    if not xdpcHandler.initialize():
        xdpcHandler.cleanup()
        exit(-1)

    xdpcHandler.scanForDots()
    if len(xdpcHandler.detectedDots()) == 0:
        print("No Movella DOT device(s) found. Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    xdpcHandler.connectDots()

    if len(xdpcHandler.connectedDots()) == 0:
        print("Could not connect to any Movella DOT device(s). Aborting.")
        xdpcHandler.cleanup()
        exit(-1)

    for device in xdpcHandler.connectedDots():
        filterProfiles = device.getAvailableFilterProfiles()
        print("Available filter profiles:")
        for f in filterProfiles:
            print(f.label())

        print(f"Current profile: {device.onboardFilterProfile().label()}")
        if device.setOnboardFilterProfile("General"):
            print("Successfully set profile to General")
        else:
            print("Setting filter profile failed!")

        print("Setting quaternion CSV output")
        device.setLogOptions(movelladot_pc_sdk.XsLogOptions_Quaternion)

        logFileName = "logfile_" + device.bluetoothAddress().replace(':', '-') + ".csv"
        print(f"Enable logging to: {logFileName}")
        if not device.enableLogging(logFileName):
            print(f"Failed to enable logging. Reason: {device.lastResultText()}")

        print("Putting device into measurement mode.")
        if not device.startMeasurement(movelladot_pc_sdk.XsPayloadMode_ExtendedEuler):
            print(f"Could not put device into measurement mode. Reason: {device.lastResultText()}")
            continue

    print("\nMain loop. Recording data for 10 seconds.")
    print("-----------------------------------------")

    # First printing some headers so we see which data belongs to which device
    s = ""
    for device in xdpcHandler.connectedDots():
        s += f"{device.bluetoothAddress():42}"
    print("%s" % s, flush=True)


    # Load inference model
    model = tf.keras.models.load_model(MODEL_NAME)
    # Buffer holds only the latest records to use as the current window
    WINDOW_SIZE = 100
    data_buffer = deque(maxlen=WINDOW_SIZE) # Window size of 100
    
    def reset_heading():
        print("\n", end="", flush=True)
        for device in xdpcHandler.connectedDots():
            print(f"\nResetting heading for device {device.portInfo().bluetoothAddress()}: ", end="", flush=True)
            if device.resetOrientation(movelladot_pc_sdk.XRM_Heading):
                print("OK\n",)
            else:
                print(f"NOK: {device.lastResultText()}\n")

    

    def update_output(data_buffer, value_dict):
        input_data = np.array([data_buffer])
        input_data = np.transpose(input_data, (0, 2, 1))  # Rearrange to be compatible with model
        prediction = model.predict(input_data, verbose = 0)
        pred_action = all_classes[prediction.argmax()]

        threshold = 0.5
        binary_pred = np.array(prediction > threshold).astype(int)
        pred_labels = mlb.inverse_transform(binary_pred)

        value_dict["pred_action"] = pred_action
        value_dict["pred_labels"] = pred_labels
        # Put the predicted action into the queue, remove an item if the queue is full
        try:
            blockingQueue.put_nowait(value_dict)
        except queue.Full:
            blockingQueue.get_nowait()  # Remove an item from the front
            blockingQueue.put_nowait(value_dict)
    

    # Setup the keyboard input listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    orientationResetDone = False
    startTime = movelladot_pc_sdk.XsTimeStamp_nowMs()
    # while movelladot_pc_sdk.XsTimeStamp_nowMs() - startTime <= 10000:
    while not endStream:
        if xdpcHandler.packetsAvailable():
            s = ""
            for device in xdpcHandler.connectedDots():
                # Retrieve a packet
                packet = xdpcHandler.getNextPacket(device.portInfo().bluetoothAddress())

                if packet.containsOrientation():
                    euler = packet.orientationEuler()
                    s += f"Roll:{euler.x():7.2f}, Pitch:{euler.y():7.2f}, Yaw:{euler.z():7.2f}| "
                if packet.containsFreeAcceleration():
                    acc = packet.freeAcceleration()
                    s += f"X:{acc[0]:7.2f}, Y:{acc[1]:7.2f}, Z:{acc[2]:7.2f}| "
                print("%s\r" % s, end="", flush=True)

                value_dict = {
                    "euler": {'x': euler.x(), 'y': euler.y(), 'z': euler.z()},
                    "free_acc": acc 
                }

                quat = packet.orientationQuaternion()
                acc = packet.freeAcceleration()
                dataLine = np.concatenate([quat,acc])
                data_buffer.append(dataLine) # Removes oldest entry if full
                if(len(data_buffer) == WINDOW_SIZE):
                    update_output(data_buffer, value_dict)


            if not orientationResetDone and movelladot_pc_sdk.XsTimeStamp_nowMs() - startTime > 5000:
                for device in xdpcHandler.connectedDots():
                    print(f"\nResetting heading for device {device.portInfo().bluetoothAddress()}: ", end="", flush=True)
                    if device.resetOrientation(movelladot_pc_sdk.XRM_Heading):
                        print("OK", end="", flush=True)
                    else:
                        print(f"NOK: {device.lastResultText()}", end="", flush=True)
                print("\n", end="", flush=True)
                orientationResetDone = True
            if keyboard_scanner.is_pressed('space'):
                reset_heading()  

    # listener.join()
    # listener_reset.join()
    print("\n-----------------------------------------", end="", flush=True)

    for device in xdpcHandler.connectedDots():
        print(f"\nResetting heading to default for device {device.portInfo().bluetoothAddress()}: ", end="", flush=True)
        if device.resetOrientation(movelladot_pc_sdk.XRM_DefaultAlignment):
            print("OK", end="", flush=True)
        else:
            print(f"NOK: {device.lastResultText()}", end="", flush=True)
    print("\n", end="", flush=True)

    print("\nStopping measurement...")
    for device in xdpcHandler.connectedDots():
        if not device.stopMeasurement():
            print("Failed to stop measurement.")
        if not device.disableLogging():
            print("Failed to disable logging.")

    xdpcHandler.cleanup()


def gui_main():


    # Check queue for action every 100ms
    def after_callback():
        try:
            # print(blockingQueue.qsize())
            message = blockingQueue.get(block=False)
        except queue.Empty:
            # let's try again later
            window.after(100, after_callback)
            return

        if message is not None:
            # we're not done yet, let's do something with the message and
            # come back later
            labels = list(message["pred_labels"][0])
            if("None" in labels):
                labels.remove("None")
            if(len(labels) == 0):
                labels.append("None")
            print(labels,"\n")
            output_label['text'] = " | ".join(labels)
            output_label['bg'] = color_dict[labels[0]]
            euler = message['euler']
            value_label['text'] = f"Roll:{euler['x']:7.2f}, Pitch:{euler['y']:7.2f}, Yaw:{euler['z']:7.2f}"
            window.after(100, after_callback)

    window = tk.Tk()
    window.title("Model Output")

    # Create a label to display the model's output
    output_label = Label(window, text="", font=("Helvetica", 24))
    output_label.pack(fill="both", expand=True, pady=50)
    output_label['text'] = 'Initialising'
    output_label['bg'] = color_dict['Initialising']

    value_label = tk.Label(window, text = f"Roll:{0:7.2f}, Pitch:{0:7.2f}, Yaw:{0:7.2f}", font=("Helvetica", 16))
    value_label.pack(side="bottom", pady=10)

    # Run the data collection on a new thread
    data_thread = threading.Thread(target=collect_data)
    data_thread.start()

    window.after(100, after_callback)
    window.geometry("600x300")
    window.mainloop()

    # Wait for threads to finish
    data_thread.join()

def main():
    collect_data()

if __name__ == "__main__":
    gui_main()
    # main()
