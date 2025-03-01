"""
Simple Visualization GUI

This script provides a minimal GUI for running visualization scripts on CSV files.
It uses only Tkinter from the Python standard library.

Usage:
    python visualization_gui.py

Author: AI Assistant
Created: Current date
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess

class VisualizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualization Tool")
        self.root.geometry("600x350")
        
        # Get base directory
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # List of available visualization scripts
        self.visualization_scripts = [
            "data_quality_viz.py",
            "time_series_analysis.py",
            "outlier_detection.py",
            "stationarity_analysis.py",
            "feature_relationships.py",
            "temporal_patterns.py"
        ]
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Cryptocurrency Data Visualization Tool", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Script selection section
        script_frame = ttk.LabelFrame(main_frame, text="Select Visualization", padding="10")
        script_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(script_frame, text="Visualization Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.script_var = tk.StringVar()
        script_dropdown = ttk.Combobox(
            script_frame, 
            textvariable=self.script_var,
            values=self.visualization_scripts,
            width=40,
            state="readonly"
        )
        script_dropdown.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        script_dropdown.current(0)  # Set default selection
        
        # CSV file selection section
        file_frame = ttk.LabelFrame(main_frame, text="Select Data File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=40)
        file_entry.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        
        browse_button = ttk.Button(file_frame, text="Browse...", command=self.browse_file)
        browse_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Run button
        run_button = ttk.Button(main_frame, text="Run Visualization", command=self.run_visualization)
        run_button.pack(pady=20)
        
        # Status message
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var, wraplength=550)
        status_label.pack(fill=tk.X, pady=5)
        
    def browse_file(self):
        """Open file dialog to select a CSV file"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.file_path_var.set(filepath)
    
    def run_visualization(self):
        """Run the selected visualization script on the chosen CSV file"""
        # Get selected script and file
        script = self.script_var.get()
        file_path = self.file_path_var.get()
        
        # Validate selections
        if not script:
            self.status_var.set("Error: Please select a visualization type")
            return
        
        if not file_path:
            self.status_var.set("Error: Please select a CSV file")
            return
        
        if not os.path.exists(file_path):
            self.status_var.set(f"Error: File not found: {file_path}")
            return
        
        # Construct absolute script path
        script_path = os.path.join(self.base_dir, "visualizations", script)
        
        if not os.path.exists(script_path):
            self.status_var.set(f"Error: Visualization script not found at: {script_path}")
            return
        
        # Run the script
        self.status_var.set(f"Running {script} on {os.path.basename(file_path)}...")
        self.root.update()
        
        try:
            # Use subprocess to run the script - print the command for debugging
            cmd = [sys.executable, script_path, file_path]
            self.status_var.set(f"Executing: {' '.join(cmd)}")
            self.root.update()
            
            # Use subprocess to run the script
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.status_var.set(f"Success! Visualizations saved to 'visualization_output' directory.\nOutput: {stdout[:100]}")
            else:
                error_msg = stderr.strip() if stderr else "Unknown error occurred"
                self.status_var.set(f"Error (code {process.returncode}): {error_msg}")
        
        except Exception as e:
            self.status_var.set(f"Exception occurred: {str(e)}")

def main():
    """Main function to create and run the GUI"""
    root = tk.Tk()
    app = VisualizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 