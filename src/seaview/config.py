"""Configuration management for the seaview processor.

This module handles loading and managing configuration settings using
Dynaconf. Settings are loaded from multiple locations in order of
increasing priority:

1. Global settings (/etc/seaview/)
2. User settings (~/.config/seaview/)
3. Current directory settings (./)
4. Environment variable specified file (SEAVIEW_SETTINGS_FILE_FOR_DYNACONF)

Attributes
----------
ENV : str
    Default environment name.
USER_DIR : pathlib.Path
    Path to user configuration directory.
GLOB_DIR : pathlib.Path
    Path to global configuration directory.
CURR_DIR : pathlib.Path
    Path to current working directory.
settings : Dynaconf
    The Dynaconf settings object with loaded configuration.
"""
import os
import pathlib

from dynaconf import Dynaconf

USER_DIR = pathlib.Path("~/.config/seaview").expanduser()
USER_DIR.mkdir(parents=True, exist_ok=True)
GLOB_DIR = pathlib.Path("/etc/seaview/")
CURR_DIR = pathlib.Path("./").absolute()
settings_files = [
    GLOB_DIR / "settings.toml",
    GLOB_DIR / ".secrets.toml",
    USER_DIR / "settings.toml",
    USER_DIR / ".secrets.toml",
    CURR_DIR / "settings.toml",
    CURR_DIR / ".secrets.toml"
    ]
extra_file = os.getenv("SEAVIEW_SETTINGS_FILE_FOR_DYNACONF")
if extra_file:
    settings_files.append(pathlib.Path(extra_file).absolute())

settings = Dynaconf(
    merge_enabled = True,
    envvar_prefix="SEAVIEW",
    DEBUG_LEVEL_FOR_DYNACONF='DEBUG',
    settings_files=settings_files,
    #secrets=[
    #    "/etc/seaview/.secrets.toml",
    #    "~/.config/seaview/.secrets.toml",
    #    "./.seaview.toml",
    #],
    environments=True,
    load_dotenv=True,
)

def change_env(new_env):
    """Change the active Dynaconf environment.

    Parameters
    ----------
    new_env : str
        The environment name to switch to (e.g., 'development', 'production').
    """
    settings.setenv(new_env)
    settings.reload()


#print("Loaded files:", settings.loaded_files)
#print("\nSettings:", dict(settings))
