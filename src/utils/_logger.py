import logging
import sys

from colorama import Fore
from tqdm.auto import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.INFO):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class ColorFormatter(logging.Formatter):
    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = Fore.YELLOW + format_orig + Fore.RESET

        elif record.levelno >= logging.WARNING:
            self._style._fmt = Fore.RED + format_orig + Fore.RESET

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result
