"""A tool to bridge natural psychology witth  the artificial."""

__version__ = "0.3.0"
from .staging import factory_settings
from .staging.scanner_cards import (
    ExpCard,
    ExpCardInit,
    save_expcard,
    load_expcard,
)
from .session_tunnel import SessionTunnel
from .datasets.prompts import parser
from . import parsers

from .scanner_models.scanner_model import ScannerModel
from .scanner_models.psyscan_io import to_csv, concat_csv
from .simulation_model.simulation_model import (
    SimulationModel,
    TaskSimulationModel,
    TrialSimulationModel,
    TrialInfoModel,
    InputSimulationModel,
    PredSimulationModel,
)
from .templates.tasks.get_task_template import get_task_template
from .feedback import FeedbackBase
from .agents import CustomAgent, ScanningAgent

__all__ = [
    "ExpCard",
    "ExpCardInit",
    "save_expcard",
    "load_expcard",
    "factory_settings",
    "ScannerModel",
    "to_csv",
    "concat_csv",
    "parser","parsers","SessionTunnel",
    "SimulationModel",
    "TaskSimulationModel",
    "TrialSimulationModel",
    "TrialInfoModel",
    "InputSimulationModel",
    "PredSimulationModel",
    "get_task_template",
    "FeedbackBase",
    "CustomAgent",
    "ScanningAgent",
]
