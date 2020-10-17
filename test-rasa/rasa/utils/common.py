import logging
import os
import shutil
import warnings
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Text, Type

import rasa.core.utils
import rasa.utils.io
from rasa.cli import utils
from rasa.cli.utils import bcolors
from rasa.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_LEVEL_LIBRARIES,
    ENV_LOG_LEVEL,
    ENV_LOG_LEVEL_LIBRARIES,
    GLOBAL_USER_CONFIG_PATH,
)

logger = logging.getLogger(__name__)


class TempDirectoryPath(str):
    """Represents a path to an temporary directory. When used as a context
    manager, it erases the contents of the directory on exit.

    """

    def __enter__(self) -> "TempDirectoryPath":
        return self

    def __exit__(
        self,
        _exc: Optional[Type[BaseException]],
        _value: Optional[Exception],
        _tb: Optional[TracebackType],
    ) -> bool:
        if os.path.exists(self):
            shutil.rmtree(self)


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())


def read_global_config() -> Dict[Text, Any]:
    """Read global Rasa configuration."""
    # noinspection PyBroadException
    try:
        return rasa.utils.io.read_config_file(GLOBAL_USER_CONFIG_PATH)
    except Exception:
        # if things go south we pretend there is no config
        return {}


def set_log_level(log_level: Optional[int] = None):
    """Set log level of Rasa and Tensorflow either to the provided log level or
    to the log level specified in the environment variable 'LOG_LEVEL'. If none is set
    a default log level will be used."""
    import logging

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger("rasa").setLevel(log_level)

    update_tensorflow_log_level()
    update_asyncio_log_level()
    update_apscheduler_log_level()
    update_socketio_log_level()

    os.environ[ENV_LOG_LEVEL] = logging.getLevelName(log_level)


def update_apscheduler_log_level() -> None:
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    apscheduler_loggers = [
        "apscheduler",
        "apscheduler.scheduler",
        "apscheduler.executors",
        "apscheduler.executors.default",
    ]

    for logger_name in apscheduler_loggers:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = False


def update_socketio_log_level() -> None:
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    socketio_loggers = ["websockets.protocol", "engineio.server", "socketio.server"]

    for logger_name in socketio_loggers:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = False


def update_tensorflow_log_level() -> None:
    """Set the log level of Tensorflow to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""

    # Disables libvinfer, tensorRT, cuda, AVX2 and FMA warnings (CPU support). This variable needs to be set before the
    # first import since some warnings are raised on the first import.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    if log_level == "DEBUG":
        tf_log_level = tf.compat.v1.logging.DEBUG
    elif log_level == "INFO":
        tf_log_level = tf.compat.v1.logging.INFO
    elif log_level == "WARNING":
        tf_log_level = tf.compat.v1.logging.WARN
    else:
        tf_log_level = tf.compat.v1.logging.ERROR

    tf.compat.v1.logging.set_verbosity(tf_log_level)
    logging.getLogger("tensorflow").propagate = False


def update_sanic_log_level(log_file: Optional[Text] = None):
    """Set the log level of sanic loggers to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""
    from sanic.log import logger, error_logger, access_logger

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    logger.setLevel(log_level)
    error_logger.setLevel(log_level)
    access_logger.setLevel(log_level)

    logger.propagate = False
    error_logger.propagate = False
    access_logger.propagate = False

    if log_file is not None:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        error_logger.addHandler(file_handler)
        access_logger.addHandler(file_handler)


def update_asyncio_log_level() -> None:
    """Set the log level of asyncio to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)
    logging.getLogger("asyncio").setLevel(log_level)


def set_log_and_warnings_filters() -> None:
    """
    Set log filters on the root logger, and duplicate filters for warnings.

    Filters only propagate on handlers, not loggers.
    """
    for handler in logging.getLogger().handlers:
        handler.addFilter(RepeatedLogFilter())

    warnings.filterwarnings("once", category=UserWarning)


def obtain_verbosity() -> int:
    """Returns a verbosity level according to the set log level."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    verbosity = 0
    if log_level == "DEBUG":
        verbosity = 2
    if log_level == "INFO":
        verbosity = 1

    return verbosity


def is_logging_disabled() -> bool:
    """Returns true, if log level is set to WARNING or ERROR, false otherwise."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    return log_level == "ERROR" or log_level == "WARNING"


def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]:
    """Sorts a list of dictionaries by their first key."""
    return sorted(dicts, key=lambda d: list(d.keys())[0])


# noinspection PyUnresolvedReferences
def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Any:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        module = globals().get(module_path, locals().get(module_path))
        if module is not None:
            return module

        if lookup_path:
            # last resort: try to import the class from the lookup path
            m = importlib.import_module(lookup_path)
            return getattr(m, module_path)
        else:
            raise ImportError(f"Cannot retrieve class from path {module_path}.")


def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.
        excluded_keys: Keys to exclude from the result.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """

    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }


def write_global_config_value(name: Text, value: Any) -> None:
    """Read global Rasa configuration."""

    try:
        os.makedirs(os.path.dirname(GLOBAL_USER_CONFIG_PATH), exist_ok=True)

        c = read_global_config()
        c[name] = value
        rasa.core.utils.dump_obj_as_yaml_to_file(GLOBAL_USER_CONFIG_PATH, c)
    except Exception as e:
        logger.warning(f"Failed to write global config. Error: {e}. Skipping.")


def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any:
    """Read a value from the global Rasa configuration."""

    def not_found():
        if unavailable_ok:
            return None
        else:
            raise ValueError(f"Configuration '{name}' key not found.")

    if not os.path.exists(GLOBAL_USER_CONFIG_PATH):
        return not_found()

    c = read_global_config()

    if name in c:
        return c[name]
    else:
        return not_found()


def mark_as_experimental_feature(feature_name: Text) -> None:
    """Warns users that they are using an experimental feature."""

    logger.warning(
        f"The {feature_name} is currently experimental and might change or be "
        "removed in the future 🔬 Please share your feedback on it in the "
        "forum (https://forum.rasa.com) to help us make this feature "
        "ready for production."
    )


def lazy_property(function: Callable) -> Any:
    """Allows to avoid recomputing a property over and over.

    The result gets stored in a local var. Computation of the property
    will happen once, on the first call of the property. All
    succeeding calls will use the value stored in the private property."""

    attr_name = "_lazy_" + function.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)

    return _lazyprop


def raise_warning(
    message: Text,
    category: Optional[Type[Warning]] = None,
    docs: Optional[Text] = None,
    **kwargs: Any,
) -> None:
    """Emit a `warnings.warn` with sensible defaults and a colored warning msg."""

    original_formatter = warnings.formatwarning

    def should_show_source_line() -> bool:
        if "stacklevel" not in kwargs:
            if category == UserWarning or category is None:
                return False
            if category == FutureWarning:
                return False
        return True

    def formatwarning(
        message: Text,
        category: Optional[Type[Warning]],
        filename: Text,
        lineno: Optional[int],
        line: Optional[Text] = None,
    ):
        """Function to format a warning the standard way."""

        if not should_show_source_line():
            if docs:
                line = f"More info at {docs}"
            else:
                line = ""

        formatted_message = original_formatter(
            message, category, filename, lineno, line
        )
        return utils.wrap_with_color(formatted_message, color=bcolors.WARNING)

    if "stacklevel" not in kwargs:
        # try to set useful defaults for the most common warning categories
        if category == DeprecationWarning:
            kwargs["stacklevel"] = 3
        elif category == UserWarning:
            kwargs["stacklevel"] = 2
        elif category == FutureWarning:
            kwargs["stacklevel"] = 3

    warnings.formatwarning = formatwarning
    warnings.warn(message, category=category, **kwargs)
    warnings.formatwarning = original_formatter


class RepeatedLogFilter(logging.Filter):
    """
    Filter repeated log records.
    """

    last_log = None

    def filter(self, record):
        current_log = (
            record.levelno,
            record.pathname,
            record.lineno,
            record.msg,
            record.args,
        )
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False
