#!/usr/bin/env python3

from mp_assignment.image import Image
from mp_assignment import __main__

import inspect
import sys
import textwrap


def main(functions):
    for function in functions:
        try:
            src = inspect.getsource(getattr(Image, function))
        except AttributeError:
            src = inspect.getsource(getattr(__main__, function))
        src = textwrap.dedent(src)
        print(src)


if __name__ == "__main__":
    main(sys.argv[1:])
