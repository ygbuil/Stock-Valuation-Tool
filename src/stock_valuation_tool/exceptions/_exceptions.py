"""Custom exceptions."""


class YahooFinanceError(Exception):
    """Error with the Yahoo Finance API."""

    def __init__(self, msg: None | str = None) -> None:
        """Provide the error message or return default.

        Args:
            self: Own class.
            msg: Custom error message. Defaults to None.
        """
        super().__init__(msg or "Something went wrong retrieving Yahoo Finance.")


class UnsortedError(Exception):
    """Error with data sorting."""

    def __init__(self, msg: None | str = None) -> None:
        """Provide the error message or return default.

        Args:
            self: Own class.
            msg: Custom error message. Defaults to None.
        """
        super().__init__(msg or "The data is not sorted as expected.")


class InvalidOptionError(Exception):
    """Provided an invalid option."""

    def __init__(self, msg: None | str = None) -> None:
        """Provide the error message or return default.

        Args:
            self: Own class.
            msg: Custom error message. Defaults to None.
        """
        super().__init__(msg or "The option provided is not valid.")


class InvalidInputDataError(Exception):
    """Provided an invalid option."""

    def __init__(self, msg: None | str = None) -> None:
        """Provide the error message or return default.

        Args:
            self: Own class.
            msg: Custom error message. Defaults to None.
        """
        super().__init__(
            msg
            or "Can not run ExponentialModel with the provided data due to a negative squared root."
        )


class InvalidModelError(Exception):
    """Provided an invalid option."""

    def __init__(self, msg: None | str = None) -> None:
        """Provide the error message or return default.

        Args:
            self: Own class.
            msg: Custom error message. Defaults to None.
        """
        super().__init__(msg or "Wrong model config.")
