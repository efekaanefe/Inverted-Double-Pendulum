import dearpygui.dearpygui as dpg
import serial
import threading
import time
import atexit

# Configure serial port
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=1)  # Replace with your port
max_log_lines = 20  # Set the maximum number of log lines
max_plot_points = 100  # Maximum number of points to display in the plot

# Store plot data for four separate series
plot_data_1 = []
plot_data_2 = []
plot_data_3 = []
plot_data_4 = []

def send_command():
    """Send a string command to the Arduino."""
    command = dpg.get_value("Command Input")
    if command:
        arduino.write((command + '\n').encode())
        add_log_entry(f"Sent: {command}")
        dpg.set_value("Command Input", "")

def read_data():
    """Continuously read data from the Arduino, parse and plot."""
    global plot_data_1, plot_data_2, plot_data_3, plot_data_4
    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            try:
                # Parse 4 comma-separated values
                values = list(map(float, data.split(",")))
                if len(values) == 4:
                    # Append each value to its respective plot data
                    plot_data_1.append(values[0])
                    plot_data_2.append(values[1])
                    plot_data_3.append(values[2])
                    plot_data_4.append(values[3])
                    
                    # Limit the number of points in each plot
                    if len(plot_data_1) > max_plot_points:
                        plot_data_1.pop(0)
                        plot_data_2.pop(0)
                        plot_data_3.pop(0)
                        plot_data_4.pop(0)
                    
                    # Update the plots and adjust axes
                    update_plots()
                    adjust_plot_axes()
                    add_log_entry(f"Received: {values}")
                else:
                    add_log_entry(f"Invalid data (expected 4 values): {data}")
            except ValueError:
                add_log_entry(f"Invalid data: {data}")
        time.sleep(0.01)

def add_log_entry(entry):
    """Add a new log entry and ensure the log has a fixed number of lines."""
    dpg.add_text(entry, parent="Log Window")
    # Check and remove old lines if exceeding the max_log_lines limit
    children = dpg.get_item_children("Log Window", 1)  # Get all child items in the Log Window
    if len(children) > max_log_lines:
        dpg.delete_item(children[0])  # Delete the oldest line

def update_plots():
    """Update the plots with the latest data."""
    dpg.set_value("plot_series_1", [list(range(len(plot_data_1))), plot_data_1])
    dpg.set_value("plot_series_2", [list(range(len(plot_data_2))), plot_data_2])
    dpg.set_value("plot_series_3", [list(range(len(plot_data_3))), plot_data_3])
    dpg.set_value("plot_series_4", [list(range(len(plot_data_4))), plot_data_4])

def adjust_plot_axes():
    """Adjust the axes dynamically based on the current data."""
    all_data = plot_data_1 + plot_data_2 + plot_data_3 + plot_data_4
    if all_data:
        min_y = min(all_data) - 5  # Add some padding
        max_y = max(all_data) + 5  # Add some padding
        dpg.set_axis_limits("y_axis", min_y, max_y)
    
    # Adjust x-axis to match the number of data points
    max_x = len(plot_data_1)
    dpg.set_axis_limits("x_axis", 0, max_x)

def close_serial():
    """Close the serial port on exit."""
    if arduino.is_open:
        arduino.close()
        print("Serial port closed.")

# Register the cleanup function to be called on exit
atexit.register(close_serial)

# Start a separate thread for reading data
thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# Create GUI layout
dpg.create_context()
dpg.create_viewport(title='Arduino Communication with Plot', width=800, height=600)

with dpg.window(label="Arduino Communication", width=800, height=600):
    # Input and Send button area
    with dpg.group(horizontal=True):
        dpg.add_text("Enter command:")
        dpg.add_input_text(tag="Command Input", width=300)
        dpg.add_button(label="Send", callback=send_command)
    dpg.add_separator()

    # Log Window to display sent and received data
    with dpg.child_window(label="Log Window", tag="Log Window", width=-1, height=200, border=False):
        pass

    # Real-time plot
    dpg.add_separator()
    with dpg.plot(label="Real-Time Data Plot", height=300, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Samples", tag="x_axis")
        with dpg.plot_axis(dpg.mvYAxis, label="Values", tag="y_axis"):
            # Add 4 line series for the 4 values
            dpg.add_line_series([], [], label="Value 1", tag="plot_series_1")
            dpg.add_line_series([], [], label="Value 2", tag="plot_series_2")
            dpg.add_line_series([], [], label="Value 3", tag="plot_series_3")
            dpg.add_line_series([], [], label="Value 4", tag="plot_series_4")

# Ensure the serial port is closed when the app exits
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_exit_callback(close_serial)
dpg.start_dearpygui()
dpg.destroy_context()
