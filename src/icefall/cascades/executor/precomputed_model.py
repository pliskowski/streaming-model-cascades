import pandas as pd


class PrecomputedModelExecutor:
    def __init__(
        self, data: pd.DataFrame, input_columns: list[str], output_columns: list[str]
    ) -> None:
        """
        Initialize the PrecomputedModelExecutor with data and column names.
        Args:
            data (pd.DataFrame): The DataFrame containing the data.
            input_columns (list[str]): The list of input column names.
            output_columns (list[str]): The list of output column names.
        """

        if not all(col in data.columns for col in input_columns):
            raise ValueError("Some input columns are not present in the DataFrame.")
        if not all(col in data.columns for col in output_columns):
            raise ValueError("Some output columns are not present in the DataFrame.")
        self.data = data.copy().set_index(input_columns, drop=True)
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.rows_retrieved = 0

    def execute(self, input_data: pd.DataFrame, selections: list[int]) -> pd.DataFrame:
        """
        Execute the PrecomputedModelExecutor on the input data and selections.

        Args:
            input_data (pd.DataFrame): The DataFrame containing the input data.
            selections (list[int]): The list of indices to select.

        Returns:
            pd.DataFrame: A DataFrame containing the output data.
        """

        if not all(col in input_data.columns for col in self.input_columns):
            raise ValueError("Some input columns are not present in the input DataFrame.")
        if not selections:
            return pd.DataFrame(index=input_data.index, columns=self.output_columns)

        if len(self.input_columns) == 1:
            keys = input_data.iloc[selections, input_data.columns.get_loc(self.input_columns[0])]
            looked_up = self.data.loc[keys][self.output_columns]
        else:
            selected_input = input_data.iloc[selections][self.input_columns]
            keys = [tuple(row) for row in selected_input.values]
            looked_up = self.data.loc[keys][self.output_columns]

        result = pd.DataFrame(index=input_data.index, columns=self.output_columns, dtype=object)
        result.iloc[selections] = looked_up.values

        self.rows_retrieved += len(selections)
        return result

    def get_rows_retrieved(self) -> int:
        """
        Get the number of rows retrieved by the executor.

        Returns:
            int: The number of rows retrieved.
        """
        return self.rows_retrieved

    def reset_rows_retrieved(self) -> None:
        """
        Reset the number of rows retrieved by the executor.
        """
        self.rows_retrieved = 0

    def __copy__(self) -> "PrecomputedModelExecutor":
        """
        Singleton, no copy
        """
        return self

    def __deepcopy__(self, memo: dict) -> "PrecomputedModelExecutor":
        """
        Singleton, no deepcopy
        """
        return self
