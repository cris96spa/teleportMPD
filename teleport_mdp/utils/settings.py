from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class YamlBaseSettings(BaseSettings):
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources = (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

        yaml_path = cls.model_config.get("yaml_file", None)
        if yaml_path:
            sources += (
                YamlConfigSettingsSource(
                    settings_cls=settings_cls,
                    yaml_file=yaml_path,
                    yaml_file_encoding=cls.model_config.get(
                        "yaml_file_encoding", "utf-8"
                    ),
                ),
            )
        return sources


class GlobalSettings(YamlBaseSettings):
    log_level: str
    model_config = SettingsConfigDict(
        yaml_file="configs/global.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
    )


class SubSettings1(BaseModel):
    name: str
    value: int


class SubSettings2(BaseModel):
    description: str
    enabled: bool


class MySettings(YamlBaseSettings):
    sub_setting1: SubSettings1
    sub_setting2: SubSettings2

    model_config = SettingsConfigDict(
        yaml_file="configs/my_settings.yaml",
        case_sensitive=False,
        extra="allow",
        yaml_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSource(settings_cls, "configs/my_settings.yaml"),
        )


if __name__ == "__main__":
    import os

    print("Working dir:", os.getcwd())
    print("YAML exists:", os.path.exists("configs/my_settings.yaml"))
    settings = MySettings()
    global_settings = GlobalSettings()
    pass
