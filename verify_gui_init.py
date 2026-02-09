import sys
import os
import tkinter as tk
# Add current directory to path so imports work
sys.path.insert(0, os.getcwd())

from modular_inspection_integrated.gui import InspectorApp

try:
    print("Initializing InspectorApp...")
    app = InspectorApp()
    app.update() # Process pending events
    print("InspectorApp initialized successfully.")
    app.destroy()
    print("InspectorApp destroyed.")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
