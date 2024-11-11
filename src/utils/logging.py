import logging


class Logger:
    """
    A Logger class to encapsulate logging setup.

    Attributes:
        logger (logging.Logger): The logger instance.

    Methods:
        get_logger(): Returns the configured logger instance.

    Usage example:
        >>> logger = Logger().get_logger()
        >>> logger.info("This is an info message")
    """

    def __init__(self, name=__name__, level=logging.DEBUG):
        """
        Initializes the Logger with a specified name and logging level.

        Args:
            name (str): The name of the logger. Defaults to __name__.
            level (int): The logging level. Defaults to logging.DEBUG.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler and set level
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create formatter and add it to the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        # Add the handler to the logger
        self.logger.addHandler(ch)

    def get_logger(self):
        """
        Returns the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger
