"""
This module disables PyTorch watchdog to avoid conflicts with Streamlit.

PyTorch creates a watchdog thread that can interfere with Streamlit's
file watchers and cause unexpected behavior, including crashes and 
excessive CPU usage.
"""

import logging
import threading
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_torch_loaded():
    """Check if torch is loaded in the current environment"""
    return 'torch' in sys.modules

def disable_torch_watchdog():
    """Disable the torch watchdog thread to avoid conflicts with Streamlit"""
    if not is_torch_loaded():
        logger.info("PyTorch not loaded, no need to disable watchdog")
        return False
    
    try:
        # Import torch if it's already in sys.modules
        import torch
        
        # Find and stop the watchdog
        for thread in threading.enumerate():
            if thread.name == 'Watchdog':
                logger.info("Found PyTorch watchdog thread, disabling...")
                
                # Use an undocumented method to stop the watchdog
                # This is a bit risky but necessary for compatibility
                if hasattr(thread, '_Thread__stop'):
                    thread._Thread__stop()
                    logger.info("PyTorch watchdog thread disabled")
                    return True
        
        logger.info("No PyTorch watchdog thread found")
        return False
    except Exception as e:
        logger.error(f"Error disabling PyTorch watchdog: {e}")
        return False

# Automatically disable watchdog when this module is imported
disable_torch_watchdog()