import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    _config = {}

    @staticmethod
    def load(config_path: str = "config/default.yaml") -> Dict[str, Any]:
        # Handle path resolution
        if not os.path.exists(config_path):
            # Try relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, config_path)
            if os.path.exists(candidate):
                config_path = candidate
            else:
                # Try relative to run dir/neuro_v1
                candidate = os.path.join(os.getcwd(), 'neuro_v1', config_path)
                if os.path.exists(candidate):
                    config_path = candidate

        try:
            with open(config_path, "r") as f:
                ConfigLoader._config = yaml.safe_load(f)
            print(f"[Config] Loaded from {config_path}")
        except FileNotFoundError:
            print(f"[Config] Warning: Config file {config_path} not found. Using empty config.")
            ConfigLoader._config = {}
            
        return ConfigLoader._config

    @staticmethod
    def get() -> Dict[str, Any]:
        if not ConfigLoader._config:
            return ConfigLoader.load()
        return ConfigLoader._config
