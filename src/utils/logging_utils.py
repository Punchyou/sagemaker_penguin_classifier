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

    _logger: None = None

    @staticmethod
    def get_logger(name=__name__, level=logging.DEBUG):
        """
        Returns the configured logger instance.

        Args:
            name (str): The name of the logger. Defaults to __name__.
            level (int): The logging level. Defaults to logging.DEBUG.

        Returns:
            logging.Logger: The configured logger instance.
        """
        if Logger._logger is None:
            Logger._logger = logging.getLogger(name)
            Logger._logger.setLevel(level)

            # Check if the logger already has handlers
            if not Logger._logger.handlers:
                # Create console handler and set level
                ch = logging.StreamHandler()
                ch.setLevel(level)

                # Create formatter and add it to the handler
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                ch.setFormatter(formatter)

                # Add the handler to the logger
                Logger._logger.addHandler(ch)

        return Logger._logger
