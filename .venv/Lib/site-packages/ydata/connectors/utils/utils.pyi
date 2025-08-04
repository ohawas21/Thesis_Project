from _typeshed import Incomplete
from collections.abc import Generator

def get_from_env(keys):
    """Returns an environment variable from one of the list of keys.

    Args:
        keys: list(str). list of keys to check in the environment
    Returns:
        str | None
    """
def is_protected_type(obj):
    """A check for preserving a type as-is when passed to
    force_text(strings_only=True)."""
def get_files_in_current_directory(path) -> Generator[Incomplete]:
    """Gets all the files under a certain path.

    Args:
        path: `str`. The path to traverse for collecting files.
    Returns:
         list of files collected under the path.
    """
def append_basename(path, filename):
    """Adds the basename of the filename to the path.

    Args:
        path: `str`. The path to append the basename to.
        filename: `str`. The filename to extract the base name from.
    Returns:
         str
    """
def check_dirname_exists(path, is_dir: bool = False) -> None: ...
def create_tmp(): ...
