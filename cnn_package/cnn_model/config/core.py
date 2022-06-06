from pathlib import Path
from typing import Dict, List, Sequence
from pydantic import BaseModel
from strictyaml import YAML, load

import cnn_model

# Project Directories
PACKAGE_ROOT = Path(cnn_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

with open(PACKAGE_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_name: str
    model_name: str
    classes_name: str
    encoder_name: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    random_state: int


class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data),
    )

    return _config


config = create_and_validate_config()

MODEL_PATH = f'{TRAINED_MODEL_DIR}/{config.app_config.model_name}_{__version__}.h5'
PIPELINE_PATH = f'{TRAINED_MODEL_DIR}/{config.app_config.pipeline_name}_{__version__}.pkl'
CLASSES_PATH = f'{TRAINED_MODEL_DIR}/{config.app_config.classes_name}_{__version__}.pkl'
ENCODER_PATH = f'{TRAINED_MODEL_DIR}/{config.app_config.encoder_name}_{__version__}.pkl'

DATA_FOLDER = f'{DATASET_DIR}/mask_dataset'
