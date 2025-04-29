"""
Create this as 'disable_torch_watch.py' in your project directory
"""

# Apply this patch before importing streamlit or torch
import sys
import types
from functools import wraps

# Find streamlit in installed packages
try:
    import streamlit
    import importlib

    # Path to the file we need to patch
    watcher_path = 'streamlit.watcher.local_sources_watcher'
    
    # Import the module
    if watcher_path in sys.modules:
        watcher_module = sys.modules[watcher_path]
    else:
        watcher_module = importlib.import_module(watcher_path)
    
    # Keep reference to the original function
    original_get_module_paths = watcher_module.get_module_paths
    
    # Create a patched version that skips torch
    @wraps(original_get_module_paths)
    def patched_get_module_paths(module):
        # Skip processing for torch modules
        if hasattr(module, "__name__") and module.__name__.startswith('torch'):
            return []
        
        # Use the original function for all other modules
        return original_get_module_paths(module)
    
    # Apply our patch
    watcher_module.get_module_paths = patched_get_module_paths
    
    print("Successfully patched Streamlit to ignore torch modules")
    
except Exception as e:
    print(f"Warning: Could not patch Streamlit watcher: {e}")