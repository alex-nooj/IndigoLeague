import importlib
import typing


def dynamic_import(target: str) -> typing.Callable:
    target_path = ".".join(target.split(".")[:-1])
    target_module = importlib.import_module(target_path)
    module = getattr(target_module, target.split(".")[-1])
    return module
