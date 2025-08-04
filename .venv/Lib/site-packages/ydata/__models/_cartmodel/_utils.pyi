from ydata.utils.data_types import DataType as DataType

def generate_function_map(base_methods: dict, method_to_enum: dict, enabled_datatypes: list[DataType], datatype_to_function: dict) -> dict:
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
