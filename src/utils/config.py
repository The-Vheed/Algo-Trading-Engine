import yaml


class ConfigManager:
    @staticmethod
    def load_config(path: str) -> dict:
        """Loads and validates the YAML configuration file."""
        with open(path, "r") as f:
            # In a production system, add validation against a schema here.
            return yaml.safe_load(f)
