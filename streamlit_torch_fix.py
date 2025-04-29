import sys
import types

# This is a monkey patch to prevent the torch._classes error
class PathMock:
    _path = []

# Only apply if torch is installed
try:
    import torch
    if hasattr(torch, "_classes"):
        # Create a mock __path__ attribute
        if not hasattr(torch._classes, "__path__"):
            torch._classes.__path__ = PathMock()
        
        # Make __getattr__ safer
        original_getattr = torch._classes.__getattr__
        
        def safe_getattr(self, attr):
            if attr == "__path__":
                return PathMock()
            return original_getattr(self, attr)
        
        torch._classes.__getattr__ = types.MethodType(safe_getattr, torch._classes)
        
except ImportError:
    pass  # torch not installed

