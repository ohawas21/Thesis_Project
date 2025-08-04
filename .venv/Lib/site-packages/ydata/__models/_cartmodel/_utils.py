from typing import Dict, List

from ydata.utils.data_types import DataType


def generate_function_map(
    base_methods: Dict,
    method_to_enum: Dict,
    enabled_datatypes: List[DataType],
    datatype_to_function: Dict,
) -> Dict:
    """Generate the map between a list method for cart model and the function
    depending on the data type.

    Args:
        base_methods (Dict[str, BaseMethods]): base methods to consider.
        method_to_enum (Dict): Mapping between a method name and its enum value.
        enabled_datatypes (List[DataType]): List of enabled datatypes
        datatype_to_function: (Dict): Map between datatypes and final functions.

    Returns:
        Dict: Mapping between a method, a type and a function (Method object)
    """
    map_type_to_function = {}
    for base_name, m in base_methods.items():
        name = method_to_enum[base_name.upper()]
        map_type_to_function[name] = {k: m.function for k in enabled_datatypes}

        # Override with DataType specific rules
        map_type_to_function[name].update(datatype_to_function.get(m, {}))

        # Check that there is a valid function for all enabled datatypes
        for d in enabled_datatypes:
            if (
                not (d in map_type_to_function[name])
                or map_type_to_function[name][d] is None
            ):
                raise Exception(
                    f"BaseMethod {m} is not defined for data type {d}!")

    return map_type_to_function
