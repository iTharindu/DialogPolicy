import asyncio
import logging
import typing
from typing import Optional, Text

from rasa.cli.utils import print_success
from rasa.core.interpreter import INTENT_MESSAGE_PREFIX, RegexInterpreter
from rasa.nlu.model import Interpreter
from rasa.nlu.utils import json_to_string

if typing.TYPE_CHECKING:
    from rasa.nlu.components import ComponentBuilder

logger = logging.getLogger(__name__)


def run_cmdline(
    model_path: Text, component_builder: Optional["ComponentBuilder"] = None
) -> None:
    interpreter = Interpreter.load(model_path, component_builder)
    regex_interpreter = RegexInterpreter()

    print_success("NLU model loaded. Type a message and press enter to parse it.")
    while True:
        print_success("Next message:")
        message = input().strip()
        if message.startswith(INTENT_MESSAGE_PREFIX):
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(regex_interpreter.parse(message))
        else:
            result = interpreter.parse(message)

        print(json_to_string(result))


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.run` directly is no longer supported. "
        "Please use `rasa run` to start a Rasa server or `rasa shell` to use your "
        "NLU model to interpret text via the command line."
    )
