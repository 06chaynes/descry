"""The standard streams descry writes to."""

import sys


def write_utf8(*streams):
    """Write ``streams`` (stdout and stderr by default) as UTF-8: answers and
    logs carry any character source has, and a pipe on Windows would take
    the locale's code page, which cannot write them all."""
    for stream in streams or (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
