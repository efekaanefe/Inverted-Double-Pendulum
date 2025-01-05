import dearpygui.dearpygui as dpg
import serial
import threading
import time
import atexit

# Configure serial port
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)  # Replace with your port
max_log_lines = 5  # Set the maximum number of log lines

def send_command():
    """Send a string command to the Arduino."""
    command = dpg.get_value("Command Input")
    if command:
        arduino.write((command + '\n').encode())
        add_log_entry(f"Sent: {command}")
        dpg.set_value("Command Input", "")

def read_data():
    """Continuously read data from the Arduino and display it in the log."""
    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            add_log_entry(f"Received: {data}")
        time.sleep(0.1)

def add_log_entry(entry):
    """Add a new log entry and ensure the log has a fixed number of lines."""
    dpg.add_text(entry, parent="Log Window")
    # Check and remove old lines if exceeding the max_log_lines limit
    children = dpg.get_item_children("Log Window", 1)  # Get all child items in the Log Window
    if len(children) > max_log_lines:
        dpg.delete_item(children[0])  # Delete the oldest line

def close_serial():
    """Close the serial port on exit."""
    if arduino.is_open:
        arduino.close()
        print("Serial port closed.")

# Ensure the serial port is closed when the app exits
atexit.register(close_serial)

# Start a separate thread for reading data
thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# Create GUI layout
dpg.create_context()
dpg.create_viewport(title='Arduino Communication', width=600, height=400)

with dpg.window(label="Arduino Communication", width=600, height=400):
    # Input and Send button area
    with dpg.group(horizontal=True):
        dpg.add_text("Enter command:")
        dpg.add_input_text(tag="Command Input", width=300)
        dpg.add_button(label="Send", callback=send_command)
    dpg.add_separator()
    
    # Log Window to display sent and received data
    with dpg.child_window(label="Log Window", tag="Log Window", width=-1, height=-1, border=False):
        pass

# Start Dear PyGui
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
