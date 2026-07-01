import unittest

import pandas as pd

from icefall.cascades.executor.precomputed_model import PrecomputedModelExecutor


class TestPrecomputedModelExecutor(unittest.TestCase):
    def setUp(self):
        self.sample_data = pd.DataFrame(
            {
                "input1": ["A", "B", "C", "D"],
                "input2": [1, 2, 3, 4],
                "output1": ["X", "Y", "Z", "W"],
                "output2": [10, 20, 30, 40],
            }
        )

    def test_initialization_valid(self):
        # Test valid initialization
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )
        self.assertEqual(executor.input_columns, ["input1", "input2"])
        self.assertEqual(executor.output_columns, ["output1", "output2"])
        self.assertIsInstance(executor.data, pd.DataFrame)

    def test_initialization_invalid_input_columns(self):
        # Test with non-existent input columns
        with self.assertRaises(ValueError):
            PrecomputedModelExecutor(
                data=self.sample_data,
                input_columns=["input1", "non_existent"],
                output_columns=["output1", "output2"],
            )

    def test_initialization_invalid_output_columns(self):
        # Test with non-existent output columns
        with self.assertRaises(ValueError):
            PrecomputedModelExecutor(
                data=self.sample_data,
                input_columns=["input1", "input2"],
                output_columns=["output1", "non_existent"],
            )

    def test_execute_valid(self):
        # Test execute with valid input data
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame({"input1": ["A", "B"], "input2": [1, 2]})

        # Process all rows
        result = executor.execute(input_data, selections=[0, 1])
        self.assertEqual(list(result.columns), ["output1", "output2"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["output1"], "X")
        self.assertEqual(result.iloc[0]["output2"], 10)
        self.assertEqual(result.iloc[1]["output1"], "Y")
        self.assertEqual(result.iloc[1]["output2"], 20)

    def test_execute_with_partial_selections(self):
        # Test execute with partial selections
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame({"input1": ["A", "B", "C"], "input2": [1, 2, 3]})

        # Process only first and third row
        result = executor.execute(input_data, selections=[0, 2])
        self.assertEqual(list(result.columns), ["output1", "output2"])
        self.assertEqual(len(result), 3)

        # First row should have values
        self.assertEqual(result.iloc[0]["output1"], "X")
        self.assertEqual(result.iloc[0]["output2"], 10)

        # Second row should be NaN (not in selections)
        self.assertTrue(pd.isna(result.iloc[1]["output1"]))
        self.assertTrue(pd.isna(result.iloc[1]["output2"]))

        # Third row should have values
        self.assertEqual(result.iloc[2]["output1"], "Z")
        self.assertEqual(result.iloc[2]["output2"], 30)

    def test_execute_with_empty_selections(self):
        # Test with empty selections
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame({"input1": ["A", "B"], "input2": [1, 2]})

        # No rows selected for processing
        result = executor.execute(input_data, selections=[])
        self.assertEqual(list(result.columns), ["output1", "output2"])
        self.assertEqual(len(result), 2)

        # All rows should be NaN
        self.assertTrue(pd.isna(result.iloc[0]["output1"]))
        self.assertTrue(pd.isna(result.iloc[0]["output2"]))
        self.assertTrue(pd.isna(result.iloc[1]["output1"]))
        self.assertTrue(pd.isna(result.iloc[1]["output2"]))

    def test_execute_missing_input_column(self):
        # Test execute with missing input column
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame(
            {
                "input1": ["A", "B"]
                # Missing 'input2'
            }
        )

        with self.assertRaises(ValueError):
            executor.execute(input_data, selections=[0, 1])

    def test_execute_not_found_inputs(self):
        # Test execute with inputs not found in the original data
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame(
            {"input1": ["E", "F"], "input2": [5, 6]}  # Not in original data
        )

        # This will likely raise a KeyError when trying to access non-existent keys
        with self.assertRaises(KeyError):
            executor.execute(input_data, selections=[0, 1])

    def test_execute_mixed_inputs(self):
        # Test with mix of found and not found inputs
        executor = PrecomputedModelExecutor(
            data=self.sample_data,
            input_columns=["input1", "input2"],
            output_columns=["output1", "output2"],
        )

        input_data = pd.DataFrame(
            {"input1": ["A", "E"], "input2": [1, 5]}  # "A" exists, "E" doesn't
        )

        # This will process first row, and raise error on second row
        with self.assertRaises(KeyError):
            executor.execute(input_data, selections=[0, 1])

        # Only processing the first row should work fine
        result = executor.execute(input_data, selections=[0])
        self.assertEqual(list(result.columns), ["output1", "output2"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["output1"], "X")
        self.assertEqual(result.iloc[0]["output2"], 10)
        self.assertTrue(pd.isna(result.iloc[1]["output1"]))
        self.assertTrue(pd.isna(result.iloc[1]["output2"]))


if __name__ == "__main__":
    unittest.main()
