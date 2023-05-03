import pathlib
import typing
import warnings

from omegaconf import OmegaConf


def load_config(
    file_path: typing.Union[str, pathlib.Path], load_cli: typing.Optional[bool] = True
) -> typing.Dict[str, typing.Any]:
    """Function for loading configs from a yaml file.

    Args:
        file_path: Path to the default file path.
        load_cli: Whether to load the command line arguments.

    Returns:
        DictConfig: Merged dict of all the command line, file, and default arguments.
    """

    if load_cli:
        cli_cfg = OmegaConf.from_cli()
    else:
        cli_cfg = OmegaConf.create({})

    default_cfg = OmegaConf.load(file_path)

    file_cfg = OmegaConf.create({})
    if "config" in cli_cfg:
        if pathlib.Path(cli_cfg.config).is_file():
            file_cfg = OmegaConf.load(cli_cfg.config)
        else:
            warnings.warn(
                "Config file was specified but does not exist. Ignoring it and using defaults..."
            )

    # Hierarchy for args is cli_cfg, then file_cfg, then default_cfg.
    return OmegaConf.to_container(OmegaConf.merge(default_cfg, file_cfg, cli_cfg))
