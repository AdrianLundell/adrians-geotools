import tkinter as tk 
import tkinter.ttk as ttk 
import os 

class FileSelecter(tk.Frame):
    """Custom tkinter widget for selecting a file through txt entry or a dialog button with validation"""

    file_formats : list
    external_var : tk.StringVar
    internal_var : tk.StringVar
    file_entry : ttk.Entry
    file_button : ttk.Button

    def __init__(self, master, external_var, external_epoch_var, file_formats, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        self.file_formats = file_formats
        self.external_var = external_var
        self.external_epoch = external_epoch_var

        self.internal_var = tk.StringVar(self, self.external_var.get())
        self.internal_epoch_var = tk.StringVar(self, value = self.external_epoch.get())

        self.file_entry = ttk.Entry(self, textvariable = self.internal_var, width=65)
        self.file_entry.bind("<FocusOut>", self.file_change, add = "+")
        self.file_entry.bind("<Tab>", self.file_change, add = "+")
        self.file_entry.bind("<Return>", self.file_change, add = "+")

        self.file_button = ttk.Button(self, command = self.file_dialog, text="Open")
        self.epoch_entry = ttk.Entry(self, textvariable=self.internal_epoch_var, width = 6)
        self.epoch_entry.bind("<FocusOut>", self.epoch_change, add = "+")
        self.epoch_entry.bind("<Tab>", self.epoch_change, add = "+")
        self.epoch_entry.bind("<Return>", self.epoch_change, add = "+")

        self.place_elements()

    def place_elements(self):
        """Geometry management"""
        self.file_entry.grid(row=0, column=0, sticky="EW")
        self.file_button.grid(row=0, column=1, sticky="W", padx=10)
        #self.epoch_entry.grid(row=0, column=2, sticky="EW")
        
        self.columnconfigure(0, weight=1)

    def file_dialog(self, *args):
        """Opens a file dialog window and updates the interval value"""
        new_file_path = tk.filedialog.askopenfilename(title="Select EOP-file", 
                                            filetypes=[("TRF-files", self.file_formats)])
                                            #initialdir = self.intital_dir)
        self.internal_var.set(new_file_path)
        self.file_change()
    
    def file_change(self, *args):
        """Updates the external var if the current path is valid, else reverts the internal var"""
        new_file_path = self.internal_var.get() 
        
        if self.validate(new_file_path):
            self.external_var.set(new_file_path)
        else:
            old_file_path = self.external_var.get()
            self.internal_var.set(old_file_path)

    def validate(self, file_path):
        """Checks that the file path is valid and is not already selected"""
        existing_path = os.path.exists(file_path)
        same_path = (file_path == self.external_var.get())
        
        name, format = os.path.splitext(file_path)
        correct_format = (format.lower() in self.file_formats)
        
        return not same_path and existing_path and correct_format

    def epoch_change(self, *args):
        """Changes the epoch value, reverting to the old value on error"""
        value = self.internal_epoch_var.get()

        try:
            value = float(value)
            self.external_epoch_var.set(value)
        except:
            value = self.external_epoch_var.get()
            self.internal_epoch_var.set(value)