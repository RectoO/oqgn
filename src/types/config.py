from typing import TypedDict, Dict


class Config(TypedDict):
    mapping: Dict[str, Dict[str, str | Dict[str, str]]]
    default_values: Dict[str, float]
    day_offsets: Dict[str, int]
