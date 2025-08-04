from __future__ import annotations
import json
from sys import getsizeof


def _camelcased(value: str) -> str:
  capitalized = ''.join(x.capitalize() for x in value.split('_'))
  return capitalized[0].lower() + capitalized[1:]


# pylint: disable=too-many-arguments, import-outside-toplevel
class FabricError(Exception):
  context: dict[str, str] | None
  description: str
  http_code: int = 500
  name: str | None
  return_value: int

  def __init__(
      self,
      context: dict[str, str] | None = None,
      description: str | None = None,
      http_code: int | None = None,
      name: str = None,
      return_value: int | None = None,
      **kwargs):
    self.context = context
    self.name = name

    if description:
      self.description = description
    if http_code or kwargs.get('httpCode', None):
      self.http_code = http_code or kwargs.get('httpCode', None)
    if return_value or kwargs.get('returnValue', None):
      self.return_value = return_value or kwargs.get('returnValue', None)

  def __iter__(self):
    yield from {
      "context": self.context,
      "name": self.name if self.name else self.__class__.__name__,
      "description": self.description,
      "http_code": self.http_code,
      "return_value": self.return_value
    }.items()

  def __repr__(self) -> str:
    from pprint import pformat
    return f"{self.__class__.__name__}(\n{pformat(dict(self), width=120, sort_dicts=False)}\n)"

  def __str__(self) -> str:
    from pprint import pformat
    string = f"{self.name} - {self.description}"
    if self.context:
      string += f"\ncontext: {pformat(self.context, width=160, sort_dicts=False)}"
    return string

  def __sizeof__(self) -> int:
    context_size = getsizeof(self.context) if self.context else 0
    http_code_size = getsizeof(str(self.http_code)) if self.http_code else 0
    return_value_size = getsizeof(str(self.return_value)) if self.return_value else 0

    return context_size \
      + getsizeof(self.description) \
      + http_code_size \
      + getsizeof(self.name) \
      + return_value_size

  def json(self):
    return json.dumps(self, default=lambda o: {_camelcased(k): v for k, v in dict(o).items()}, ensure_ascii=False)
