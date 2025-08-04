from ydata.constraints.base import Constraint
from ydata.dataset.dataset import Dataset

class ConstraintEngine:
    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        """Create a ConstraintEngine to validate constraints against a Dataset.

        Args:
            constraints (List[Constraint]): List of constraints

        Returns:
            ConstraintEngine: ConstraintEngine instance
        """
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a single constraint to the engine.

        Args:
            constraint (Constraint): A constraint
        """
    def add_constraints(self, constraints: list[Constraint]) -> None:
        """Add some constraint to the engine.

        Args:
            constraints (List[Constraint]): List of constraints
        """
    def validate(self, dataset: Dataset) -> None:
        """Validate the engine against a dataset.

        Args:
            dataset (Dataset): Dataset against which to validate the constraints
        """
    def violated_constraints(self, constraint: Constraint | str) -> dict: ...
    def violated_rows(self, constraint: Constraint | str) -> dict:
        """Return the violated rows report for a particular constraint,
        including the violated rows mask.

        Args:
            constraint (Union[Constraint, str]): Constraint

        Returns:
            dict: violated rows report
        """
    def violated_columns(self, constraint: Constraint | str) -> dict:
        """Return the violated columns report for a particular constraint.

        Args:
            constraint (Union[Constraint, str]): Constraint

        Returns:
            dict: violated columns report
        """
    def summary(self, include_rows: bool = False) -> dict:
        """Return the constraint summary for a validated constraint engine
        instance.

        Args:
            include_rows (bool): If True, includes the violated rows mask

        Returns:
            dict: violated rows summary
        """
    def explain_constraints(self) -> dict[str, str]:
        """Explain the constraints in the engine using the constraint
        representation.

        Returns:
            dict[str, str]: constraint explanation
        """
