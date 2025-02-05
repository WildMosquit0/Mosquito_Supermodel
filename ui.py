import tkinter as tk
from tkinter import filedialog, messagebox
import yaml
import os
import subprocess
import threading
import sys

CONFIG_FILE = "config.yaml"
CONFIG_DIR = "configs"
CONFIG_FILES = {
    "infer": os.path.join(CONFIG_DIR, "infer.yaml"),
    "analyze": os.path.join(CONFIG_DIR, "analyze.yaml"),
}

def load_global_config():
    """Load global configuration (main.py path)."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)
    return {"main_path": "main.py"}  # Default path

def save_global_config():
    """Save global configuration (main.py path)."""
    global_config["main_path"] = main_path_entry.get()
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(global_config, file, default_flow_style=False)
    messagebox.showinfo("Success", "Main script path saved!")

def select_main_path():
    """Select path to main.py."""
    path = filedialog.askopenfilename(title="Select main.py", filetypes=[("Python Files", "*.py")])
    if path:
        main_path_entry.delete(0, tk.END)
        main_path_entry.insert(0, path)

def load_config(task_name):
    """Load configuration from the selected YAML file."""
    config_file = CONFIG_FILES.get(task_name)
    if not os.path.exists(config_file):
        messagebox.showerror("Error", f"Config file not found: {config_file}")
        return {}
    
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def save_config(task_name):
    """Save modified configuration to the YAML file with correct data types and formatting."""
    config_file = CONFIG_FILES.get(task_name)
    if not config_file:
        return

    config = {}
    for key, entry in entries.items():
        if isinstance(entry, dict):
            config[key] = {subkey: convert_value(subentry.get()) for subkey, subentry in entry.items()}
        else:
            config[key] = convert_value(entry.get())

    with open(config_file, "w") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False, indent=4, allow_unicode=True)

    messagebox.showinfo("Success", f"Configuration saved to {config_file}")

def convert_value(value):
    """Convert string input into correct data types (int, float, bool, or string)."""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.isdigit():
        return int(value)
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        return value  # Keep as string if not a number or boolean


def update_config_ui():
    """Update the UI when the task selection changes."""
    task_name = task_var.get()
    config = load_config(task_name)

    for widget in config_frame.winfo_children():
        widget.destroy()

    global entries
    entries = {}

    for key, value in config.items():
        if isinstance(value, dict):  # Handle nested dictionaries
            tk.Label(config_frame, text=f"{key}:", font=("Arial", 10, "bold")).pack(pady=5)
            sub_entries = {}
            for subkey, subvalue in value.items():
                tk.Label(config_frame, text=f"  {subkey}:").pack()
                sub_entry = tk.Entry(config_frame, width=50)
                sub_entry.insert(0, str(subvalue))
                sub_entry.pack()
                sub_entries[subkey] = sub_entry
            entries[key] = sub_entries  # Store nested structure
        else:
            tk.Label(config_frame, text=f"{key}:").pack()
            entry = tk.Entry(config_frame, width=50)
            entry.insert(0, str(value))
            entry.pack()
            entries[key] = entry

    save_button.pack(pady=5)
    run_button.pack(pady=5)
    status_label.pack(pady=5)

def run_script():
    """Run main.py with the selected task and update status."""
    task_name = task_var.get()
    main_path = main_path_entry.get()

    if not os.path.exists(main_path):
        messagebox.showerror("Error", f"main.py not found at: {main_path}")
        return

    status_label.config(text="Running...", fg="blue")
    root.update_idletasks()

    def execute():
        try:
            abs_main_path = os.path.abspath(main_path)
            python_exec = sys.executable  # Get current Python executable
            
            # Debug print
            print(f"Running: {python_exec} {abs_main_path} --task_name {task_name}")

            # Method 1: Using subprocess.Popen (Captures and prints live output)
            process = subprocess.Popen([python_exec, abs_main_path, "--task_name", task_name],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
            for line in iter(process.stdout.readline, ''):
                print(line, end="")
            process.stdout.close()
            process.wait()

            if process.returncode == 0:
                print("Execution successful.")
                status_label.config(text="Completed! Closing...", fg="green")
                root.update_idletasks()
                root.after(1500, root.destroy)  # Close GUI after 1.5 seconds
            else:
                error_message = process.stderr.read()
                print("Execution failed:", error_message)
                status_label.config(text="Error! Check script logs.", fg="red")
                messagebox.showerror("Error", f"main.py failed: {error_message}")

        except Exception as e:
            print("Unexpected error:", str(e))
            status_label.config(text="Error! Check logs.", fg="red")
            messagebox.showerror("Error", f"Unexpected error: {e}")

    threading.Thread(target=execute, daemon=True).start()

# Load global config at startup
global_config = load_global_config()

# Create GUI Window
root = tk.Tk()
root.title("Supermodel")
root.geometry("500x900")

# Select Main Path
tk.Label(root, text="Main Script Path:").pack(pady=5)
main_path_entry = tk.Entry(root, width=50)
main_path_entry.insert(0, global_config.get("main_path", "main.py"))
main_path_entry.pack()
tk.Button(root, text="Browse", command=select_main_path).pack(pady=5)
tk.Button(root, text="Save Path", command=save_global_config).pack(pady=5)

# Task Selection
task_var = tk.StringVar(value="analyze")
tk.Label(root, text="Select Task:").pack(pady=5)
tk.Radiobutton(root, text="Infer", variable=task_var, value="infer", command=update_config_ui).pack()
tk.Radiobutton(root, text="Analyze", variable=task_var, value="analyze", command=update_config_ui).pack()

# Config Frame (where YAML fields appear)
config_frame = tk.Frame(root)
config_frame.pack(pady=10)

# Status Label
status_label = tk.Label(root, text="", fg="black")

# Buttons
save_button = tk.Button(root, text="Save Config", command=lambda: save_config(task_var.get()))
run_button = tk.Button(root, text="Run Script", command=run_script)

update_config_ui()  # Load initial UI

root.mainloop()
