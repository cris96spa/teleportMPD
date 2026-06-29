"""Generic component registry — the factory backbone for the package.

A registry maps a normalized string key (typically an enum value such as
``Algorithm.PPO`` or ``Curriculum.STATIC``) to a factory callable, so the *selection*
of which concrete class to build lives in one declarative place rather than being
hard-coded as ``if/elif`` branches in the orchestrator. Each subclass owns its own
``_registry`` dict; the :meth:`register` decorator adds entries and :meth:`create`
looks them up.
"""

from collections.abc import Callable
from enum import Enum
from logging import getLogger
from typing import Any, ClassVar, Generic, TypeVar

logger = getLogger(__name__)

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Base registry mapping string/enum keys to factory callables.

    Subclasses **must** declare their own ``_registry`` class variable (enforced by
    :meth:`__init_subclass__`). A subclass may override ``_PASSTHROUGH_KEY`` so that a
    designated key resolves to ``None`` (e.g. the ``none`` curriculum yields no
    scheduler), mirroring the "passthrough" idiom.

    Example:
        >>> class MyRegistry(ComponentRegistry[Base]):
        ...     _registry: ClassVar[dict[str, Callable[..., Base | None]]] = {}
        >>> @MyRegistry.register("variant")
        ... def _build(**kwargs: Any) -> Base: ...
        >>> MyRegistry.create("variant", **kwargs)
    """

    _registry: ClassVar[dict[str, Callable[..., Any]]] = {}
    _PASSTHROUGH_KEY: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure every subclass gets its own ``_registry`` dict.

        Args:
            **kwargs: Forwarded to :meth:`object.__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        if "_registry" not in cls.__dict__:
            logger.warning("%s did not define _registry; creating an empty one.", cls.__name__)
            cls._registry = {}

    @classmethod
    def register(cls, key: str | Enum) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a factory function (or class) under ``key``.

        When decorating a class, it is wrapped so ``create(key, **kwargs)`` calls
        ``TheClass(**kwargs)``.

        Args:
            key: String or enum identifier for the component.

        Returns:
            A decorator that stores the target in the registry and returns it unchanged.
        """
        normalized = cls.normalize_key(key)

        def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
            if isinstance(target, type):
                cls._registry[normalized] = lambda **kw: target(**kw)
            else:
                cls._registry[normalized] = target
            logger.debug("Registered %r in %s.", normalized, cls.__name__)
            return target

        return decorator

    @classmethod
    def create(cls, key: str | Enum, **kwargs: Any) -> T | None:
        """Look up ``key`` and call the registered factory.

        Args:
            key: String or enum identifier; a passthrough key resolves to ``None``.
            **kwargs: Forwarded to the registered factory callable.

        Returns:
            A new component instance, or ``None`` for the passthrough key.

        Raises:
            ValueError: If ``key`` is not registered and is not the passthrough key.
        """
        normalized = cls.normalize_key(key)
        if cls._PASSTHROUGH_KEY is not None and normalized == cls._PASSTHROUGH_KEY:
            return None
        if normalized not in cls._registry:
            raise ValueError(
                f"Unknown key {normalized!r} for {cls.__name__}; available: {cls.get_available()}."
            )
        return cls._registry[normalized](**kwargs)

    @classmethod
    def is_registered(cls, key: str | Enum) -> bool:
        """Whether ``key`` resolves to a registered factory or the passthrough key."""
        normalized = cls.normalize_key(key)
        return normalized == cls._PASSTHROUGH_KEY or normalized in cls._registry

    @classmethod
    def get_available(cls) -> list[str]:
        """Return the sorted registered keys (excluding passthrough)."""
        return sorted(cls._registry.keys())

    @classmethod
    def normalize_key(cls, key: str | Enum) -> str:
        """Convert an enum or string key to a canonical lowercase string.

        Args:
            key: Raw key value.

        Returns:
            The normalized lowercase string.
        """
        if isinstance(key, Enum):
            return str(key.value).lower()
        return str(key).lower()
