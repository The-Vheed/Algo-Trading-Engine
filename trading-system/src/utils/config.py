from typing import Any
import yaml

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def validate_config(self, required_keys: list) -> bool:
        for key in required_keys:
            if key not in self.config:
                return False
        return True