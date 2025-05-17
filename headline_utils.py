import pickle
import logging
from pathlib import Path


def get_model_paths():
    """Get model paths with fallbacks for different directory structures"""
    # Try to find paths relative to current file first
    base_dir = Path(__file__).parent.resolve()

    # Check for common project structures
    potential_paths = [
        # Structure 1: Direct subdirectories
        {
            "processed_data": base_dir / "agentic_news_editor" / "processed_data",
            "output": base_dir / "model_output",
        },
        # Structure 2: Flat structure
        {
            "processed_data": base_dir / "processed_data",
            "output": base_dir / "model_output",
        },
        # Structure 3: Current directory
        {"processed_data": Path("processed_data"), "output": Path("model_output")},
    ]

    # Find the first valid path configuration
    for paths in potential_paths:
        if paths["output"].exists():
            return paths

    # If no valid configuration found, create directories in current working dir
    output_dir = Path("model_output")
    processed_data_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    processed_data_dir.mkdir(exist_ok=True)

    logging.warning(f"Created new directory structure in {Path().resolve()}")
    return {"processed_data": processed_data_dir, "output": output_dir}


def load_ctr_model(model_path=None, fallback_path=None):
    """Standardized model loading with consistent error handling"""
    if model_path is None:
        # Try default locations
        model_path = Path("model_output/ctr_model.pkl")

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Validate model data has required components
        required_keys = ["model", "model_name", "scaler", "encoders", "feature_names"]
        if not all(key in model_data for key in required_keys):
            raise ValueError(f"Model at {model_path} is missing required components")

        logging.info(
            f"Successfully loaded model {model_data['model_name']} from {model_path}"
        )
        return model_data, None

    except Exception as e:
        error_msg = f"Failed to load model from {model_path}: {str(e)}"
        logging.error(error_msg)

        # Try fallback if provided
        if fallback_path and fallback_path != model_path:
            logging.info(f"Attempting to load fallback model from {fallback_path}")
            try:
                return load_ctr_model(fallback_path, None)
            except:
                pass

        return None, error_msg
