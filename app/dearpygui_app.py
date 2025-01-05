import dearpygui.dearpygui as dpg
import serial
import threading
import time

# Configure serial port
# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Replace 'COM3' with your port

def send_command():
    """Send a string command to the Arduino."""
    command = dpg.get_value("Command Input")
    if command:
        arduino.write((command + '\n').encode())
        dpg.add_text(f"Sent: {command}", parent="Log Window")
        dpg.set_value("Command Input", "")

def read_data():
    """Continuously read data from the Arduino and display it in the log."""
    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            with dpg.mutex():  # Safely update the GUI from a separate thread
                dpg.add_text(f"Received: {data}", parent="Log Window")
        time.sleep(0.1)

# Start a separate thread for reading data
thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# Create GUI layout
dpg.create_context()
dpg.create_viewport(title='Arduino Communication', width=600, height=400)

with dpg.window(label="Arduino Communication", width=600, height=400):
    dpg.add_input_text(label="Enter Command", tag="Command Input", width=300)
    dpg.add_button(label="Send", callback=send_command)
    dpg.add_separator()
    # Make the Log Window fill the entire available space
    with dpg.child_window(label="Log Window", tag="Log Window", width=-1, height=-1, border=False):
        pass

# Start Dear PyGui
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
