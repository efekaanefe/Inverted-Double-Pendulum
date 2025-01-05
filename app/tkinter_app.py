import tkinter as tk
from tkinter import scrolledtext
import serial
import time
import threading

# Configure serial port
# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)  # Replace 'COM3' with your port

def send_command():
    """Send a string command to the Arduino."""
    command = command_entry.get()
    if command:
        arduino.write((command + '\n').encode())
        log_text.insert(tk.END, f"Sent: {command}\n")
        command_entry.delete(0, tk.END)

def read_data():
    """Continuously read data from the Arduino and display it in the log."""
    while True:
        data = arduino.readline().decode('utf-8').strip()
        if data:
            log_text.insert(tk.END, f"Received: {data}\n")
            log_text.see(tk.END)  # Auto-scroll to the latest message
        time.sleep(0.1)

# GUI setup
app = tk.Tk()
app.title("Arduino Serial Communication")

# Command Entry
tk.Label(app, text="Enter Command:").pack(pady=5)
command_entry = tk.Entry(app, width=40)
command_entry.pack(pady=5)

# Send Button
send_button = tk.Button(app, text="Send", command=send_command)
send_button.pack(pady=5)

# Log Text Area
log_text = scrolledtext.ScrolledText(app, width=50, height=20)
log_text.pack(pady=10)

# Start reading data in a separate thread
thread = threading.Thread(target=read_data, daemon=True)
thread.start()

# Run the app
app.mainloop()

