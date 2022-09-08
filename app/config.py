"""
Dynaconf Settings
"""
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix=False,
    settings_files=[
        'settings.yaml',
    ],
    validators=[],
)