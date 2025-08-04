import time
from typing import Dict, Union

from pandas import DataFrame as pdDataFrame
from pandas import Series as pdSeries
from pandas import concat as pdconcat

from ydata.constraints.base import ColumnConstraint, Constraint, RowConstraint
from ydata.constraints.exceptions import ConstraintDoesNotExist
from ydata.constraints.utils import require_validate
from ydata.dataset.dataset import Dataset
from ydata.utils.configuration import TextStyle


# @typechecked
class ConstraintEngine:
    def __init__(self, constraints: list[Constraint] | None = None):
        """Create a ConstraintEngine to validate constraints against a Dataset.

        Args:
            constraints (List[Constraint]): List of constraints

        Returns:
            ConstraintEngine: ConstraintEngine instance
        """
        self._constraints = {}
        self._constraint_mask = None
        self._validated = False  # Indicates if the constraints have been validated
        self._validation_time = {}

        if constraints is not None:
            self.add_constraints(constraints)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a single constraint to the engine.

        Args:
            constraint (Constraint): A constraint
        """
        if self._validated:
            self._validated = False
        self.add_constraints([constraint])

    def add_constraints(self, constraints: list[Constraint]) -> None:
        """Add some constraint to the engine.

        Args:
            constraints (List[Constraint]): List of constraints
        """
        if self._validated:
            self._validated = False
        for c in constraints:
            self._constraints[self.__get_constraint_index(c)] = c

    def validate(self, dataset: Dataset) -> None:
        """Validate the engine against a dataset.

        Args:
            dataset (Dataset): Dataset against which to validate the constraints
        """
        if not self._validated:
            self._constraint_mask = self.__validate_per_constraint(dataset)
            self._validated = True

    @staticmethod
    def __get_constraint_index(constraint: Constraint) -> str:
        """Get the constraint index to use from a constraint object.

        If the user specify the name for a constraint, use it.
        Otherwise, use its representation.

        Args:
            constraint (Constraint): A constraint

        Returns:
            str: index
        """
        index = getattr(constraint, "name", None)
        if not index:
            index = constraint.__repr__()
        return index

    def __validate_per_constraint(self, dataset: Dataset) -> pdDataFrame:
        """Get the constraint index to use from a constraint object.

        If the user specify the name for a constraint, use it.
        Otherwise, use its representation.

        Args:
            dataset (Dataset):  Dataset against which to validate the constraints

        Returns:
            pd.DataFrame: validation result where each column represent a constraint
        """
        if len(self._constraints) == 0:
            return pdDataFrame()  # No constraint, empty mask
        results = []
        # Check constraints one by one as it is the fastest option
        for i, c in self._constraints.items():
            start = time.time()
            mask = c.validate(dataset)
            mask.columns = [f"{i}|{col}" for col in mask.columns]
            end = time.time()
            self._validation_time[i] = round(end - start, 2)
            results.append(mask)
        results = pdconcat(results, axis=1)
        return results

    @require_validate
    def __violated_rows(self, mask: pdDataFrame | None = None) -> pdDataFrame:
        """Mask indicating the violated row given a constraint mask.

        A constraint mask can be multicolumn and therefore, this function aggregates the results
        to indicate only the rows that are violated.

        Args:
            mask (pd.Dataframe): Constraint mask

        Returns:
            pd.DataFrame: validation result where each column represent a constraint
        """
        row_constraints = [
            name
            for name, const in self._constraints.items()
            if isinstance(const, RowConstraint)
        ]

        if mask is None:
            mask = self._constraint_mask
        cols = [col for col in mask.columns if col.split("|")[
            0] in row_constraints]
        mask = mask[cols]

        length = len(mask.columns)
        mask = mask.fillna(False)
        mask = mask.astype(int).sum(axis=1) < length
        return mask

    def __get_constraint_key(self, constraint: Constraint | str) -> str | None:
        """Get the constraint key if it exists.

        If constraint is a string, it is assumed to be the name, otherwise, we test the index.

        Args:
            constraint (Union[Constraint, str]): Constraint

        Returns:
            Optional[str]: constraint index in the engine
        """
        if constraint in self._constraints.keys():
            return constraint
        elif self.__get_constraint_index(constraint) in self._constraints.keys():
            return self.__get_constraint_index(constraint)

    def violated_constraints(self, constraint: Constraint | str) -> dict:
        if isinstance(constraint, str):
            if constraint not in self._constraints:
                raise ConstraintDoesNotExist(
                    f"The constraint `{constraint}` does not exist"
                )
            constraint = self._constraints[constraint]

        if isinstance(constraint, RowConstraint):
            return self.violated_rows(constraint)
        return self.violated_columns(constraint)

    def violated_rows(self, constraint: Constraint | str) -> dict:
        """Return the violated rows report for a particular constraint,
        including the violated rows mask.

        Args:
            constraint (Union[Constraint, str]): Constraint

        Returns:
            dict: violated rows report
        """
        c = self.__get_constraint_key(constraint)
        if c is None:
            raise ConstraintDoesNotExist("The constraint does not exist")

        mask = self._constraint_mask
        cols = [e for e in mask.columns if e.startswith(f"{c}|")]
        cols_name = [e.split("|")
                     for e in mask.columns if e.startswith(f"{c}|")]
        v_rows = self.__violated_rows(mask[cols])
        v_rows.columns = cols_name
        result = self.__summary_constraint(v_rows)
        result["validation_time"] = self._validation_time[c]
        return result

    def violated_columns(self, constraint: Union[Constraint, str]) -> dict:
        """Return the violated columns report for a particular constraint.

        Args:
            constraint (Union[Constraint, str]): Constraint

        Returns:
            dict: violated columns report
        """
        c = self.__get_constraint_key(constraint)
        if c is None:
            raise ConstraintDoesNotExist("The constraint does not exist")

        mask = self._constraint_mask
        cols = [e for e in mask.columns if e.startswith(f"{c}|")]
        col_names = [e.split("|")[-1] for e in cols]
        # astype is necessary because of in presence of nan the series
        # the dtype will be considered object instead if bool
        mask = ~mask[cols].dropna().astype(bool)
        mask.columns = col_names
        count = mask.sum(axis=1)[0]
        ratio = float(count) / mask.shape[1] if mask.shape[1] > 0 else 0.0
        result = {
            "column_violation_count": count,
            "column_violation_ratio": ratio,
            "violated_columns": [
                col
                for col, violation in mask.to_dict(orient="records")[0].items()
                if violation
            ],
        }

        result["validation_time"] = self._validation_time[c]

        return result

    @staticmethod
    def __summary_constraint(v_rows: pdSeries) -> Dict:
        """Return a summary of violation for a given mask column.

        Args:
            v_rows (pd.DataFrame): Mask

        Returns:
            dict: violated rows summary
        """
        count = v_rows.sum()
        ratio = float(count) / v_rows.shape[0] if v_rows.shape[0] > 0 else 0.0
        result = {
            "rows_violation_count": count,
            "rows_violation_ratio": ratio,
            "rows": v_rows,
        }
        return result

    @require_validate
    def __violation_per_constraint(self, include_rows: bool = False) -> dict:
        """Return the violated rows report for all constraints.

        Args:
            include_rows (bool): If True, includes the violated rows mask

        Returns:
            dict: violated rows report
        """
        results = {}
        for c in self._constraints.keys():
            results[c] = self.violated_constraints(c)
            if not include_rows:
                if "rows" in results[c]:
                    del results[c]["rows"]
        return results

    @require_validate
    def summary(self, include_rows: bool = False) -> dict:
        """Return the constraint summary for a validated constraint engine
        instance.

        Args:
            include_rows (bool): If True, includes the violated rows mask

        Returns:
            dict: violated rows summary
        """
        if len(self._constraints) == 0:
            return {}

        # FIXME Summary counts column constraints incorrectly
        v_rows = self.__violated_rows(self._constraint_mask)
        summary = self.__summary_constraint(v_rows)
        if not include_rows:
            del summary["rows"]
        summary["violation_per_constraint"] = self.__violation_per_constraint(
            include_rows
        )
        return summary

    def explain_constraints(self) -> dict[str, str]:
        """Explain the constraints in the engine using the constraint
        representation.

        Returns:
            dict[str, str]: constraint explanation
        """
        return {k: str(v) for k, v in self._constraints.items()}

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        constraint_total = len(self._constraints)
        print_string = "\n" + TextStyle.BOLD + "Constraint Engine \n \n" + TextStyle.END
        summary_string = f'Number of Constraints: {constraint_total}\n'
        detail_string = ""

        if self._validated and constraint_total > 0:
            summary_dict = self.summary()
            constraints_violated = 0
            for constraint_name, constraint in self._constraints.items():
                individual_constraints_violated = 0
                individual_constraints_violated_ratio = 0
                constraint_summary = summary_dict['violation_per_constraint'][constraint_name]
                if isinstance(constraint, RowConstraint):
                    individual_constraints_violated += constraint_summary.get(
                        "rows_violation_count", 0)
                    individual_constraints_violated_ratio = constraint_summary.get(
                        "rows_violation_ratio", 0)
                elif isinstance(constraint, ColumnConstraint):
                    individual_constraints_violated = constraint_summary.get(
                        "column_violation_count", 0)
                    individual_constraints_violated_ratio = constraint_summary.get(
                        "column_violation_ratio", 0)

                if individual_constraints_violated > 0:
                    detail_string += f'{constraint}: {TextStyle.BOLD} {individual_constraints_violated} violated ({round(100* individual_constraints_violated_ratio, 2):.2f}%) {TextStyle.END}\n'
                else:
                    detail_string += f'{constraint}: {individual_constraints_violated} ({100* individual_constraints_violated_ratio}%)\n'

                if individual_constraints_violated > 0:
                    constraints_violated += 1

            summary_string += "Number of Constraints Violated: " + \
                f'{constraints_violated}' + \
                f' ({round(100* constraints_violated/constraint_total, 2):.2}%)\n\n'

            return print_string + summary_string + detail_string
        if bool(self._constraints):
            summary_string += "Number of Constraints Violated: Constraint Engine not validated\n\n"
            for constraint_name, constraint in self._constraints.items():
                detail_string += f'{constraint}: not validated\n'
            return print_string + summary_string + detail_string

        print_string += "No constraints have been added to the constraint engine."
        return print_string
