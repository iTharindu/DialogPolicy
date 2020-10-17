import argparse
import os
from typing import Text

from rasa.cli.utils import print_error

from rasa.nlu import training_data
from rasa.nlu.utils import write_to_file


def convert_training_data(
    data_file: Text, out_file: Text, output_format: Text, language: Text
):
    if not os.path.exists(data_file):
        print_error(
            "Data file '{}' does not exist. Provide a valid NLU data file using "
            "the '--data' argument.".format(data_file)
        )
        return

    if output_format == "json":
        td = training_data.load_data(data_file, language)
        output = td.nlu_as_json(indent=2)
    elif output_format == "md":
        td = training_data.load_data(data_file, language)
        output = td.nlu_as_markdown()
    else:
        print_error(
            "Did not recognize output format. Supported output formats: 'json' and "
            "'md'. Specify the desired output format with '--format'."
        )
        return

    write_to_file(out_file, output)


def main(args: argparse.Namespace):
    convert_training_data(args.data, args.out, args.format, args.language)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.nlu.convert` directly is "
        "no longer supported. "
        "Please use `rasa data` instead."
    )
