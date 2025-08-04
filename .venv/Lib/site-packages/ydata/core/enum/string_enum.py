from enum import Enum


class StringEnum(Enum):
  @classmethod
  def _missing_(cls, value):
    if isinstance(value, str):
      upper_value = value.upper()

      key = cls._key_from_str(upper_value)
      if key is not None:
        return key

      lower_value = value.lower()

      key = cls._key_from_str(lower_value)
      if key is not None:
        return key

    raise ValueError(f"{value} is not a valid {cls.__name__}")

  @classmethod
  def _key_from_str(cls, value: str):
    if value in cls._member_map_.keys():
      return cls._member_map_[value]

    return None

  def __hash__(self) -> int:
    return hash(self.value)

  def __eq__(self, other: object) -> bool:
    return self.value == other.value
