"""
A simpler fix for the torch._classes error in Streamlit
"""

import sys
import types

# This is a monkey patch to prevent the torch._classes error
class PathMock:
    _path = []

# Only apply if torch is installed
try:
    import torch
    
    # Create mock class for torch._classes
    class ClassesMock:
        def __init__(self, original_classes):
            self.original = original_classes
            self.__path__ = PathMock()
        
        def __getattr__(self, name):
            if name == "__path__":
                return PathMock()
            return getattr(self.original, name)
    
    # Apply the mock only if torch._classes exists
    if hasattr(torch, "_classes"):
        original_classes = torch._classes
        torch._classes = ClassesMock(original_classes)
    
except ImportError:
    pass  # torch not installed
except Exception as e:
    print(f"Warning: Failed to patch torch._classes: {e}")
    # Continue without patching to avoid breaking the application

