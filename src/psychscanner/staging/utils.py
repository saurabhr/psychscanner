import time
from datetime import datetime
from pathlib import Path
import numpy as np
import json
import ollama

import pydantic
from typing import Any

from ..datasets import *

class ticTok:
    def __init__(self):
        self.tic = time.time()

    def tok(self, format=None):
        return round(time.time() - self.tic, 4)


def structured_to_dict(obj: Any) -> Any:
    """
    Recursively converts an object with attributes (or a nested structure) into a dictionary.

    Args:
        obj (Any): The input object (can be a class instance, list, dict, tuple, etc.).

    Returns:
        Any: A dictionary representation of the object, or the object itself if it's a primitive type.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj  # Return primitive types as-is

    if isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO format string

    if isinstance(obj, dict):
        return {
            key: structured_to_dict(value) for key, value in obj.items()
        }  # Process dictionary recursively

    if isinstance(obj, list):
        return [structured_to_dict(item) for item in obj]  # Process lists recursively

    if isinstance(obj, tuple):
        return tuple(
            structured_to_dict(item) for item in obj
        )  # Process tuples recursively

    if hasattr(obj, "__dict__"):
        return {
            key: structured_to_dict(value) for key, value in obj.__dict__.items()
        }  # Convert object attributes to dict

    return obj  # Return unhandled types as-is


# gets model information and works only for ollamamodel
def get_model_card(modelname):
    """
    modelname (str): ollama modelnames
    """
    tempmodelinfo = {}
    model_card = {}

    if Path(modelname).is_dir():
        print(
            "not ollama model, return empty data", {"ollama_present": False}
        )  # any local model with dir
        return {"ollama_present": False}
    elif isinstance(modelname, str):
        all_model_list = ollama.list()
        # n_all_models = convert_models_to_dict(all_model_list)
        all_models = structured_to_dict(all_model_list)["models"]
        # print(all_models)
        # model infromation

        modelinfo = [i for i in all_models if i["model"] == modelname][
            0
        ]  # {i : j for i,j in all_models.items() if i==modelname}

        for i, j in modelinfo.items():
            if isinstance(j, str):
                tempmodelinfo[i] = j
            if isinstance(j, pydantic.types.ByteSize):
                tempmodelinfo[i] = j

        for i, j in modelinfo.items():
            if isinstance(j, dict):
                tempmodelinfo = {**tempmodelinfo, **j}

    else:
        print("----<!!!!>---- no models to run. Setting llm model as None")

    tempmodelinfo["modelname"] = modelname
    tempmodelinfo["ollama_present"] = True
    model_card = tempmodelinfo.copy()

    return model_card



