"""
Enhanced Visualization GUI

This script provides a GUI for running visualization scripts on CSV files.
It uses Tkinter from the Python standard library and supports visualizations
that now use pandas, matplotlib, and seaborn.

Features:
- CSV file dropdown with auto-detection from datasets folder
- Individual visualization running
- Aggregate visualization across multiple datasets

Usage:
    python visualization_gui.py

Author: AI Assistant
Modified: Current date
"""

import os
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess
from tkinter import messagebox
from tkinter import scrolledtext
import datetime
import threading
import queue
import glob
import pandas as pd

class VisualizationGUI:
    """GUI for running cryptocurrency data visualizations"""
    
    def __init__(self, root):
        """Initialize the GUI components"""
        self.root = root
        root.title("Cryptocurrency Data Visualization Tool")
        root.geometry("900x700")
        root.resizable(True, True)
        
        # Set styles
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Arial", 12))
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TCombobox", font=("Arial", 12))
        
        # Initialize inclusion variables for dataset directories
        self.include_processed_exchanges = tk.BooleanVar(value=True)
        self.include_additional_features = tk.BooleanVar(value=True)
        
        # Available visualization scripts
        self.available_visualizations = [
            "data_quality_viz.py - Data Quality Analysis",
            "time_series_analysis.py - Time Series Structure Analysis",
            "outlier_detection.py - Outlier Detection",
            "stationarity_analysis.py - Stationarity Analysis",
            "feature_relationships.py - Feature Relationships",
            "temporal_patterns.py - Temporal Patterns Analysis"
        ]
        
        # Queue for thread-safe logging - initialize this BEFORE finding CSV files
        self.log_queue = queue.Queue()
        
        # Main frame
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="Cryptocurrency Data Visualization Tool", 
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=10)
        
        # Description label
        desc_label = ttk.Label(
            main_frame,
            text="This tool provides beautiful visualizations using pandas, matplotlib, and seaborn.",
            font=("Arial", 10),
            wraplength=550
        )
        desc_label.pack(pady=5)
        
        # Script selection section
        script_frame = ttk.LabelFrame(main_frame, text="Select Visualization", padding="10")
        script_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(script_frame, text="Visualization Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.script_var = tk.StringVar()
        script_dropdown = ttk.Combobox(
            script_frame, 
            textvariable=self.script_var,
            values=self.available_visualizations,
            width=40,
            state="readonly"
        )
        script_dropdown.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
        script_dropdown.current(0)  # Set default selection
        
        # Add log display area - create this BEFORE finding CSV files
        log_frame = ttk.LabelFrame(main_frame, text="Output Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Add ScrolledText widget for logs
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)  # Read-only initially
        
        # Find all CSV files in the datasets folder - AFTER initializing the log system
        self.csv_files = self.find_csv_files(
            include_processed_exchanges=self.include_processed_exchanges.get(),
            include_additional_features=self.include_additional_features.get()
        )
        
        # CSV file selection section with dropdown
        file_frame = ttk.LabelFrame(main_frame, text="Select Data File", padding="10")
        file_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(file_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.file_path_var = tk.StringVar()
        
        # CSV Dropdown
        self.file_dropdown = ttk.Combobox(
            file_frame, 
            textvariable=self.file_path_var,
            values=self.csv_files,
            width=60,
            state="readonly"
        )
        self.file_dropdown.grid(row=0, column=1, sticky=tk.W+tk.E, padx=10, pady=5)
        if self.csv_files:
            self.file_dropdown.current(0)  # Set default selection
        
        # Make the column containing the dropdown expandable
        file_frame.columnconfigure(1, weight=1)
        
        # Refresh CSV list button
        refresh_button = ttk.Button(file_frame, text="Refresh", command=self.refresh_csv_files)
        refresh_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Add directory inclusion checkboxes
        inclusion_frame = ttk.Frame(file_frame)
        inclusion_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # Processed exchanges checkbox
        exchanges_checkbox = ttk.Checkbutton(
            inclusion_frame, 
            text="Include processed exchanges folder", 
            variable=self.include_processed_exchanges,
            command=self.refresh_csv_files
        )
        exchanges_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Additional features checkbox
        features_checkbox = ttk.Checkbutton(
            inclusion_frame, 
            text="Include additional features folder", 
            variable=self.include_additional_features,
            command=self.refresh_csv_files
        )
        features_checkbox.pack(side=tk.LEFT, padx=10)
        
        # Or browse manually option
        ttk.Label(file_frame, text="Or:").grid(row=3, column=0, sticky=tk.W, pady=5)
        browse_button = ttk.Button(file_frame, text="Browse for file...", command=self.browse_file)
        browse_button.grid(row=3, column=1, sticky=tk.W, padx=10, pady=5)
        
        # Run buttons
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(pady=10)
        
        run_button = ttk.Button(buttons_frame, text="Run Visualization", command=self.run_visualization)
        run_button.pack(side=tk.LEFT, padx=10)
        
        # Add aggregate visualization button
        aggregate_button = ttk.Button(
            buttons_frame, 
            text="Run Aggregate Visualization", 
            command=self.run_aggregate_visualization
        )
        aggregate_button.pack(side=tk.LEFT, padx=10)
        
        # Add combined dataset button
        combined_dataset_button = ttk.Button(
            buttons_frame, 
            text="Create Combined Dataset", 
            command=self.create_combined_dataset
        )
        combined_dataset_button.pack(side=tk.LEFT, padx=10)
        
        # Status message
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var, wraplength=550)
        status_label.pack(fill=tk.X, pady=5)
        
        # Start the log queue processor
        self.root.after(100, self.process_log_queue)
        
    def find_csv_files(self, include_processed_exchanges=True, include_additional_features=True):
        """Find all CSV files in the datasets folder
        
        Args:
            include_processed_exchanges: Whether to include files from processed_exchanges folder
            include_additional_features: Whether to include files from additional_features folder
        """
        csv_files = []
        
        # Base path for datasets
        datasets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        
        # Find all CSV files recursively
        for root, dirs, files in os.walk(datasets_path):
            # Skip processed_exchanges directory if not included
            if not include_processed_exchanges and "processed_exchanges" in root:
                continue
                
            # Skip additional_features directory if not included
            if not include_additional_features and "additional_features" in root:
                continue
                
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    # Store the full path 
                    csv_files.append(full_path)
        
        # Sort alphabetically
        csv_files.sort()
        
        if not csv_files:
            self.log("No CSV files found in datasets folder")
        else:
            self.log(f"Found {len(csv_files)} CSV files in datasets folder")
            
        return csv_files
    
    def refresh_csv_files(self):
        """Refresh the list of CSV files"""
        self.log("Refreshing CSV file list...")
        # Use the class variables for inclusion settings
        self.csv_files = self.find_csv_files(
            include_processed_exchanges=self.include_processed_exchanges.get(),
            include_additional_features=self.include_additional_features.get()
        )
        self.file_dropdown['values'] = self.csv_files
        if self.csv_files:
            self.file_dropdown.current(0)
        self.log("CSV file list refreshed")
        
    def log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_queue.put(log_message)
    
    def process_log_queue(self):
        """Process queued log messages"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)  # Auto-scroll to end
                self.log_text.config(state=tk.DISABLED)
                self.log_queue.task_done()
        except queue.Empty:
            pass
        finally:
            # Check again after 100ms
            self.root.after(100, self.process_log_queue)
            
    def browse_file(self):
        """Open file dialog to select a CSV file"""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.file_path_var.set(filepath)
            self.log(f"Selected file: {filepath}")
            
            # Add to dropdown if not already there
            if filepath not in self.csv_files:
                self.csv_files.append(filepath)
                self.file_dropdown['values'] = self.csv_files
                self.file_dropdown.set(filepath)
    
    def run_visualization(self):
        """Run the selected visualization script on the selected data file"""
        # Get selected visualization
        selected_viz = self.script_var.get()
        if not selected_viz:
            self.status_var.set("Error: Please select a visualization type")
            self.log("Error: Please select a visualization type")
            return
            
        # Extract the script name from the selection
        script_name = selected_viz.split(" - ")[0]
        visualization_name = selected_viz.split(" - ")[1]
        
        # Get selected file path
        file_path = self.file_path_var.get()
        if not file_path:
            self.status_var.set("Error: Please select a data file")
            self.log("Error: Please select a data file")
            return
            
        if not os.path.exists(file_path):
            self.status_var.set(f"Error: File not found: {file_path}")
            self.log(f"Error: File not found: {file_path}")
            return
        
        # Run the selected visualization script
        self.status_var.set(f"Running {script_name} on {file_path}...")
        self.log(f"Starting visualization: {visualization_name}")
        self.root.update()
        
        # Create output directory structure
        base_output_dir = "visualization_output"
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            self.log(f"Created base output directory: {base_output_dir}")
        
        # Create a unique output folder for this visualization run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_type = script_name.replace(".py", "")
        data_file_name = os.path.basename(file_path).replace(".csv", "")
        output_folder_name = f"{viz_type}_{data_file_name}_{timestamp}"
        output_dir = os.path.join(base_output_dir, output_folder_name)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.log(f"Created output directory: {output_dir}")
        
        # Construct the path to the script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "AI_Gen_Code", "visualizations", script_name)
        
        if not os.path.exists(script_path):
            self.status_var.set(f"Error: Visualization script not found at: {script_path}")
            self.log(f"Error: Visualization script not found at: {script_path}")
            return
        
        # Run the script in a separate thread
        self.status_var.set(f"Running visualization...")
        threading.Thread(target=self._run_script_thread, 
                        args=(script_path, file_path, output_dir, visualization_name)).start()
                        
    def run_aggregate_visualization(self):
        """Run the same visualization on multiple files for comparison"""
        self.log("Starting aggregate visualization...")
        
        # Get selected visualization
        selected_viz = self.script_var.get()
        if not selected_viz:
            self.status_var.set("Error: Please select a visualization type")
            self.log("Error: Please select a visualization type")
            return
            
        # Extract the script name from the selection
        script_name = selected_viz.split(" - ")[0]
        visualization_name = selected_viz.split(" - ")[1]
        
        # Check if there are CSV files available
        if not self.csv_files:
            self.status_var.set("Error: No CSV files found in datasets folder")
            self.log("Error: No CSV files found in datasets folder")
            return
        
        # Track all files by category for better selection
        # Getting files by category - ONLY include validated files when available
        processed_validated_files = [f for f in self.csv_files if "processed_validated" in os.path.basename(f).lower()]
        # Only include processed files WITHOUT a validated version
        processed_files = [f for f in self.csv_files if "processed" in os.path.basename(f).lower() 
                         and "validated" not in os.path.basename(f).lower()
                         and not any(os.path.basename(f).replace(".csv", "_validated.csv") in vf for vf in processed_validated_files)]
        
        # Get files from the processed_exchanges and additional_features folders
        processed_exchanges_files = [f for f in self.csv_files if "processed_exchanges" in f.lower()]
        additional_features_files = [f for f in self.csv_files if "additional_features" in f.lower()]
        
        # Display file categories in log
        self.log(f"Found {len(processed_validated_files)} validated files")
        self.log(f"Found {len(processed_files)} processed files (without validated versions)")
        self.log(f"Found {len(processed_exchanges_files)} processed exchanges files")
        self.log(f"Found {len(additional_features_files)} additional features files")
        
        # Ask user which types of files to include
        message = "Select file types to include in aggregate visualization:"
        options = ["Validated files", "Processed files", "Exchange data", "Additional features"]
        
        selected_types = messagebox.askquestion(
            "Select File Types",
            f"{message}\n\nInclude all available file types?",
            icon='question'
        )
        
        # Prepare list of files to process based on user selection
        files_to_process = []
        
        if selected_types == 'yes':
            # Use all categorized files
            files_to_process = list(set(processed_validated_files + processed_files + 
                                      processed_exchanges_files + additional_features_files))
        else:
            # Custom file type selection dialog
            custom_dialog = tk.Toplevel(self.root)
            custom_dialog.title("Select File Types")
            custom_dialog.geometry("400x350")
            custom_dialog.transient(self.root)
            custom_dialog.grab_set()
            
            # Create variables for checkboxes
            include_validated = tk.BooleanVar(value=True)
            include_processed = tk.BooleanVar(value=True)
            include_processed_exchanges = tk.BooleanVar(value=self.include_processed_exchanges.get())
            include_additional_features = tk.BooleanVar(value=self.include_additional_features.get())
            
            # Create checkboxes
            ttk.Checkbutton(custom_dialog, text=f"Validated files ({len(processed_validated_files)})", 
                          variable=include_validated).pack(anchor=tk.W, padx=20, pady=5)
            ttk.Checkbutton(custom_dialog, text=f"Processed files ({len(processed_files)})", 
                          variable=include_processed).pack(anchor=tk.W, padx=20, pady=5)
            ttk.Checkbutton(custom_dialog, text=f"Processed exchanges folder ({len(processed_exchanges_files)})", 
                          variable=include_processed_exchanges).pack(anchor=tk.W, padx=20, pady=5)
            ttk.Checkbutton(custom_dialog, text=f"Additional features folder ({len(additional_features_files)})", 
                          variable=include_additional_features).pack(anchor=tk.W, padx=20, pady=5)
            
            # Function to process selection
            def process_selection():
                nonlocal files_to_process
                if include_validated.get():
                    files_to_process.extend(processed_validated_files)
                if include_processed.get():
                    files_to_process.extend([f for f in processed_files if f not in files_to_process])
                if include_processed_exchanges.get():
                    files_to_process.extend([f for f in processed_exchanges_files if f not in files_to_process])
                if include_additional_features.get():
                    files_to_process.extend([f for f in additional_features_files if f not in files_to_process])
                custom_dialog.destroy()
            
            # Function to cancel
            def cancel_selection():
                nonlocal files_to_process
                files_to_process = []
                custom_dialog.destroy()
            
            # Add buttons
            button_frame = ttk.Frame(custom_dialog)
            button_frame.pack(fill=tk.X, pady=20)
            ttk.Button(button_frame, text="OK", command=process_selection).pack(side=tk.RIGHT, padx=10)
            ttk.Button(button_frame, text="Cancel", command=cancel_selection).pack(side=tk.RIGHT, padx=10)
            
            # Wait for dialog to close
            self.root.wait_window(custom_dialog)
        
        # Remove duplicates
        files_to_process = list(set(files_to_process))
        
        # If no files selected, cancel
        if not files_to_process:
            self.status_var.set("Aggregate visualization cancelled")
            self.log("Aggregate visualization cancelled - no files selected")
            return
        
        # Create output directory structure for aggregate visualization
        base_output_dir = "visualization_output"
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
            self.log(f"Created base output directory: {base_output_dir}")
        
        # Create a unique output folder for this aggregate visualization run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_type = script_name.replace(".py", "")
        output_folder_name = f"aggregate_{viz_type}_{timestamp}"
        aggregate_output_dir = os.path.join(base_output_dir, output_folder_name)
        
        if not os.path.exists(aggregate_output_dir):
            os.makedirs(aggregate_output_dir)
            self.log(f"Created aggregate output directory: {aggregate_output_dir}")
        
        # Construct the path to the script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "AI_Gen_Code", "visualizations", script_name)
        
        if not os.path.exists(script_path):
            self.status_var.set(f"Error: Visualization script not found at: {script_path}")
            self.log(f"Error: Visualization script not found at: {script_path}")
            return
        
        # Update status
        self.status_var.set(f"Running aggregate visualization on {len(files_to_process)} files...")
        self.log(f"Starting aggregate visualization: {visualization_name} on {len(files_to_process)} files")
        
        # Create subfolders for each CSV in the aggregate output directory
        self.log("Files to process:")
        for file_path in files_to_process:
            self.log(f"  - {os.path.basename(file_path)}")
        
        # Run the script for each file in a separate thread
        threading.Thread(target=self._run_aggregate_thread, 
                        args=(script_path, files_to_process, aggregate_output_dir, visualization_name)).start()
    
    def _run_aggregate_thread(self, script_path, files_to_process, output_dir, visualization_name):
        """Run visualization on multiple files in a separate thread"""
        self.log(f"Starting thread for aggregate visualization: {visualization_name}")
        try:
            # Check if files_to_process contains additional features
            additional_features_files = [f for f in files_to_process if "additional_features" in f.lower()]
            processed_exchanges_files = [f for f in files_to_process if "processed_exchanges" in f.lower() or "exchange" in os.path.basename(f).lower()]
            
            # Create combined output for additional features if requested
            if additional_features_files:
                self.log("Creating combined CSV file for additional features...")
                self._create_combined_csv(additional_features_files, 
                                         os.path.join(output_dir, "combined_additional_features.csv"),
                                         "additional_features")
            
            # Create combined output for exchange files if requested  
            if processed_exchanges_files:
                self.log("Creating combined CSV file for exchange data...")
                self._create_combined_csv(processed_exchanges_files, 
                                         os.path.join(output_dir, "combined_exchanges.csv"),
                                         "exchanges")
            
            # Process individual files
            for file_path in files_to_process:
                try:
                    file_name = os.path.basename(file_path)
                    self.log(f"Processing {file_name}...")
                    
                    # Create subfolder for this file's output
                    file_output_dir = os.path.join(output_dir, file_name.replace('.csv', ''))
                    os.makedirs(file_output_dir, exist_ok=True)
                    
                    # Run the visualization script
                    self._run_script(script_path, file_path, file_output_dir)
                    
                    self.log(f"Completed visualization for {file_name}")
                except Exception as e:
                    self.log(f"Error processing {file_name}: {str(e)}")
            
            self.status_var.set(f"Aggregate visualization complete!")
            self.log(f"Aggregate visualization complete!")
            self.log(f"Results saved to: {output_dir}")
            
        except Exception as e:
            self.status_var.set(f"Error in aggregate visualization: {str(e)}")
            self.log(f"Error in aggregate visualization: {str(e)}")
            
    def _create_combined_csv(self, files, output_path, data_type):
        """Create a combined CSV file from multiple data files
        
        Args:
            files: List of file paths to combine
            output_path: Path to save the combined CSV file
            data_type: Type of data being combined ('additional_features' or 'exchanges')
        """
        try:
            # Create empty DataFrame to hold combined data
            combined_df = pd.DataFrame()
            
            # Process each file
            for file_path in files:
                try:
                    file_name = os.path.basename(file_path)
                    self.log(f"Adding {file_name} to combined file...")
                    
                    # Read CSV file
                    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                    
                    # For exchange data, rename columns to include exchange name
                    if data_type == 'exchanges':
                        # Extract exchange name from filename
                        exchange_name = file_name.split('_')[0]
                        # Replace numeric indices with exchange symbols
                        for col in df.columns:
                            if col != 'Index':  # Skip the Index column itself
                                df.rename(columns={col: f"{exchange_name}_{col}"}, inplace=True)
                    
                    # For first file, use it as base
                    if combined_df.empty:
                        combined_df = df
                    else:
                        # Join on Date index
                        combined_df = combined_df.join(df, how='outer')
                    
                except Exception as e:
                    self.log(f"Error processing {file_name} for combined file: {str(e)}")
            
            # Save combined file
            if not combined_df.empty:
                combined_df.to_csv(output_path)
                self.log(f"Saved combined file with {len(combined_df.columns)} columns to {output_path}")
            else:
                self.log("No data to save to combined file")
                
        except Exception as e:
            self.log(f"Error creating combined CSV: {str(e)}")
    
    def _run_script_thread(self, script_path, file_path, output_dir, visualization_name):
        """Run the visualization script in a separate thread"""
        try:
            self.log(f"Starting thread for {visualization_name} on {os.path.basename(file_path)}")
            
            # Run the script
            self._run_script(script_path, file_path, output_dir)
            
            # Update status when done
            success_msg = f"Visualization complete: {visualization_name}"
            self.log(success_msg)
            self.root.after(0, lambda: self.status_var.set(success_msg))
        except Exception as e:
            error_msg = f"Error in visualization: {str(e)}"
            self.log(error_msg)
            self.root.after(0, lambda: self.status_var.set(error_msg))
            
    def _run_script(self, script_path, file_path, output_dir):
        """Run a visualization script using subprocess"""
        try:
            # Ensure the output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Get the file name for logging
            file_name = os.path.basename(file_path)
            
            # Prepare the command
            cmd = [sys.executable, script_path, file_path, "--output_dir", output_dir]
            self.log(f"Executing: {' '.join(cmd)}")
            
            # Run the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Process output and errors
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(f"OUTPUT ({file_name}): {line.strip()}")
            
            for line in iter(process.stderr.readline, ''):
                if line:
                    self.log(f"ERROR ({file_name}): {line.strip()}")
            
            # Clean up
            process.stdout.close()
            process.stderr.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.log(f"Successfully processed {file_name}")
                return True
            else:
                self.log(f"Error processing {file_name} (code {return_code})")
                return False
                
        except Exception as e:
            self.log(f"Exception during processing {os.path.basename(file_path)}: {str(e)}")
            return False

    def create_combined_dataset(self):
        """Create a comprehensive combined dataset for AI training"""
        self.log("Starting creation of combined dataset for AI training...")
        self.status_var.set("Creating combined dataset for AI training...")
        
        # Find the path to the create_combined_dataset.py script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "AI_Gen_Code", "data_collection", "create_combined_dataset.py")
        
        if not os.path.exists(script_path):
            error_msg = f"Error: Combined dataset script not found at: {script_path}"
            self.status_var.set(error_msg)
            self.log(error_msg)
            return
        
        # Run the script in a separate thread
        threading.Thread(target=self._run_combined_dataset_script, args=(script_path,)).start()
    
    def _run_combined_dataset_script(self, script_path):
        """Run the combined dataset script in a separate thread"""
        try:
            # Run the script
            cmd = [sys.executable, script_path]
            self.log(f"Executing: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Process output
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.log(f"COMBINED DATASET: {line.strip()}")
            
            for line in iter(process.stderr.readline, ''):
                if line:
                    self.log(f"ERROR: {line.strip()}")
            
            process.stdout.close()
            process.stderr.close()
            return_code = process.wait()
            
            if return_code == 0:
                success_msg = "Combined dataset created successfully! Ready for AI training."
                self.log(success_msg)
                self.root.after(0, lambda: self.status_var.set(success_msg))
            else:
                error_msg = f"Error (code {return_code}): Failed to create combined dataset"
                self.log(error_msg)
                self.root.after(0, lambda: self.status_var.set(error_msg))
                
        except Exception as e:
            error_msg = f"Exception creating combined dataset: {str(e)}"
            self.log(error_msg)
            self.root.after(0, lambda: self.status_var.set(error_msg))

def main():
    """Main function to create and run the GUI"""
    root = tk.Tk()
    app = VisualizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 