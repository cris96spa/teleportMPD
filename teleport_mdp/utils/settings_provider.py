from teleport_mdp.utils.settings import GlobalSettings
from teleport_mdp.utils.singleton import SingletonMeta


class SettingsProvider(meta=SingletonMeta):
    """Singleton class to provide global settings for the application."""

    def __init__(self):
        self._global_settings = GlobalSettings()

    @property
    def global_settings(self) -> GlobalSettings:
        """Get the global settings."""
        return self._global_settings
