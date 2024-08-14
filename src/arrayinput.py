import logging
from argparse import Action, ArgumentTypeError
from typing import Callable, Iterable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseArrayType(Action):
    """Parse list input types from string to python types.

    All inputs for hyperparameter search come as string of
    comma-separated values. For every such input, the string needs to
    be split into list and then the elements parsed into python
    types. This class itself parses the keeps the elements as strings
    but its subclasses may convert the strings into other python
    types. Subclass needs to define the parsing function in 'function'
    attribute such as:

    functions = (BaseArrayType.parse_int,)

    Functions are defined in an iterable because certain inputs allow
    multiple types, such as MaxFeaturesArray. For this, input like
    "None,sqrt,log2,2,1.0" (so NoneType, str, str, int, float) is
    perfectly valid. Hence for this particular class the 'functions'
    attribute would look like:

    functions = (
        BaseArrayType.parse_none,
        BaseArrayType.parse_str,
        BaseArrayType.parse_int,
        BaseArrayType.parse_float,
    )

    Parsing functions are applied in the order given.

    The scipy object does it's own input validation so there's no
    point in doing thorough validation here, like checking if the
    number is in correct range. Hence the purpose of this class and
    its subclasses is to simply parse the strings and check if they
    are of correct type. For the most part, python primitives should
    be enough. Still, if more complex validation is required here, the
    subclass can define it's own parsing function (and then define the
    parsing function list in 'get_functions' method) or even override
    the entire 'validate' method.

    """

    separator = ","
    truthy = ("true", "t")
    falsy = ("false", "f")
    nil = ("none",)
    functions: tuple[Callable, ...]
    options: tuple[str, ...]

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            parsed = self.validate(values)
        except ArgumentTypeError as exc:
            print("ArgumentTypeError error happening", exc.args)
            # parser.error(self.err_msg.format(values))
            # parser.error(f'Invalid values {values} for {self.dest}')
            parser.error(exc.args[0])

        # store the array in the namespace
        setattr(namespace, self.dest, parsed)

    def validate(self, value_str):
        return self.parse_arr(value_str)

    def get_functions(self) -> Iterable[Callable]:
        return self.functions

    def parse_arr(self, value_str):
        return [
            self.parse_value(k, self.get_functions())
            for k in value_str.split(self.separator)
            if k
        ]

    def parse_value(self, value, functions):
        found = False
        for func in functions:
            try:
                parsed = func(value)
                found = True
                break
            except ArgumentTypeError as exc:
                # error here is fine, the parsers are tried one after
                # the other and the first ones may not work. record a
                # low-level message in the logs and continue to the next
                logger.debug(exc.args[0])
                continue

        if found:
            return parsed
        else:
            raise ArgumentTypeError(f'"{value}" is not a valid value for {self.dest}')

    @staticmethod
    def parse_str(value):
        return value

    @staticmethod
    def parse_bool(value):
        if value.lower() in BaseArrayType.truthy:
            return True
        elif value.lower() in BaseArrayType.falsy:
            return False
        else:
            raise ArgumentTypeError(f'"{value}" is not a valid boolean value.')

    @staticmethod
    def parse_none(value):
        if value.lower() in BaseArrayType.nil:
            return None
        else:
            raise ArgumentTypeError(f'"{value}" is not a valid NoneType value.')

    @staticmethod
    def parse_float(value):
        try:
            return float(value)
        except ValueError as exc:
            raise ArgumentTypeError(f'"{value}" is not a valid float value.') from exc

    @staticmethod
    def parse_int(value):
        try:
            return int(value)
        except ValueError as exc:
            raise ArgumentTypeError(f'"{value}" is not a valid int value.') from exc

    @staticmethod
    def parse_number(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError as exc:
                raise ArgumentTypeError(f"{value} is not a number") from exc

    def parse_choices(self, value):
        if value in self.options:
            return value
        else:
            raise ArgumentTypeError(f"{value} is not a valid choice for {self.dest}")


class UniqueArray(BaseArrayType):
    def validate(self, value_str):
        values = super().validate(value_str)
        values_set = list(set(values))
        if len(values) > len(values_set):
            logger.warning("Non-unique values submitted for %s", self.dest)

        return values_set


class BoolArray(UniqueArray, BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of booleans."
    functions = (BaseArrayType.parse_bool,)


class FloatArray(BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of floats."
    functions = (BaseArrayType.parse_float,)


class IntArray(BaseArrayType):
    err_msg = "{} is not a valid comma-separated list of integers."
    functions = (BaseArrayType.parse_int,)


class NumberArray(BaseArrayType):
    functions = (
        BaseArrayType.parse_int,
        BaseArrayType.parse_float,
    )


class ChoiceArray(BaseArrayType):
    # choices need instance data, hence the function
    def get_functions(self) -> Iterable[Callable]:
        return (self.parse_choices,)
