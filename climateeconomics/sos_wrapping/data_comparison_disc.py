import logging

import autograd.numpy as anp
import numpy as np
import pandas as pd
from autograd import jacobian
from sostrades_core.execution_engine.sos_wrapp import SoSWrapp
from sostrades_core.tools.post_processing.charts.chart_filter import ChartFilter
from sostrades_core.tools.post_processing.charts.two_axes_instanciated_chart import (
    InstanciatedSeries,
    TwoAxesInstanciatedChart,
)


class DataComparisonDiscipline(SoSWrapp):
    _ontology_data = {
        "label": "Data Comparison Discipline",
        "type": "Research",
        "source": "",
        "version": "0.1",
        "icon": "fa-solid fa-minimize"

    }

    DESC_IN = {
        "default_interpolation_method": {
            "type": "string",
            "default": "none",
            "possible_values": ["none", "linear"],
        },
        "config_df": {
            "type": "dataframe",
            "dataframe_descriptor": {
                "source_ns": ("string", None, True),
                "variable_name": ("string", None, True),
                "column_name": ("string", None, True),
                "local_var": ("string", None, True),
                "local_column": ("string", None, True),
                "common_column": ("string", None, True),
                "error_metric": ("string", None, True),
                "weight": ("float", None, True),
                "interpolation_method": ("string", None, True),
            },
            "structuring": True,
        },
        "default_error_metric": {
            "type": "string",
            "default": "mse",
            "possible_values": ["mse", "mae", "nrmse", "rmse"],
        },
    }

    DESC_OUT = {
        "individual_errors": {
            "type": "dict",
            "description": "Error metrics for each comparison",
        },
        "overall_error": {
            "type": "float",
            "description": "Weighted overall error across comparisons",
        },
        "comparison_details": {
            "type": "dict",
            "description": "Detailed comparison results",
        },
        "jacobians": {
            "type": "dataframe",
            "description": "Jacobians of error metrics with respect to source dataframe values",
        },
    }

    def __init__(self, sos_name, logger: logging.Logger):
        super().__init__(sos_name, logger)

    def setup_sos_disciplines(self):
        """
        Dynamically add input dataframes based on configuration
        """
        if "config_df" not in self.get_data_in():
            return

        config_df = self.get_sosdisc_inputs("config_df")
        dynamic_inputs = {}

        # skip if config_df does not exist
        if config_df is None:
            return

        for _, row in config_df.iterrows():
            # Source namespace inputs
            dynamic_inputs[f"{row['variable_name']}"] = {
                "type": "dataframe",
                "visibility": SoSWrapp.SHARED_VISIBILITY,
                "namespace": row["source_ns"],
            }

            # Local namespace inputs
            local_input_name = f"{row['local_var']}"

            if local_input_name not in dynamic_inputs:
                dynamic_inputs[local_input_name] = {
                    "type": "dataframe",
                    "dataframe_descriptor": {
                        row["local_column"]: ("float", None, True)
                    },
                }
            else:
                dynamic_inputs[local_input_name]["dataframe_descriptor"][
                    row["local_column"]
                ] = ("float", None, True)

        self.add_inputs(dynamic_inputs)

    def run(self):
        config_df = self.get_sosdisc_inputs('config_df')
        error_metric = self.get_sosdisc_inputs('default_error_metric')
        input_dict = self.get_sosdisc_inputs()

        errors = []
        comparison_details = {}
        jacobians = {}

        for idx, row in config_df.iterrows():
            # try:
            source_input_name = f"{row['variable_name']}"
            local_input_name = f"{row['local_var']}"

            source_df = input_dict[source_input_name]
            local_df = input_dict[local_input_name]

            common_col = row['common_column']
            source_col = row['column_name']
            local_col = row['local_column']

            source_df_sorted = source_df.sort_values(by=common_col)
            local_df_sorted = local_df.sort_values(by=common_col)

            # Find common x range
            x_min = max(source_df_sorted[common_col].min(), local_df_sorted[common_col].min())
            x_max = min(source_df_sorted[common_col].max(), local_df_sorted[common_col].max())

            # Filter data within common range
            source_df_filtered = source_df_sorted[
                (source_df_sorted[common_col] >= x_min) & (source_df_sorted[common_col] <= x_max)]
            local_df_filtered = local_df_sorted[
                (local_df_sorted[common_col] >= x_min) & (local_df_sorted[common_col] <= x_max)]

            x1 = anp.array(source_df_filtered[common_col].values)
            y1 = anp.array(source_df_filtered[source_col].values)
            x2 = anp.array(local_df_filtered[common_col].values)
            y_true = anp.array(local_df_filtered[local_col].values)

            # Full source data for Jacobian
            x1_full = anp.array(source_df_sorted[common_col].values)
            y1_full = anp.array(source_df_sorted[source_col].values)

            interpolation_method = row.get('interpolation_method', 'linear')
            if interpolation_method.lower() != 'none':
                # Interpolation is on
                def interpolation_wrapper(y1_var):
                    return self.interpolate_data_autograd(x1, y1_var, x2)

                y_pred = interpolation_wrapper(y1)
            else:
                # Interpolation is off, use only common x values
                common_x = anp.intersect1d(x1, x2)
                mask1 = anp.isin(x1, common_x)
                mask2 = anp.isin(x2, common_x)
                y1 = y1[mask1]
                y_true = y_true[mask2]
                y_pred = y1

            def error_wrapper(y_var):
                if interpolation_method.lower() != 'none':
                    interpolated = interpolation_wrapper(y_var)
                else:
                    interpolated = y_var
                return self.compute_error_autograd(y_true, interpolated, metric=error_metric)

            error = error_wrapper(y1)

            # Compute Jacobian with respect to full source data
            def full_error_wrapper(y_full_var):
                # Extract the relevant part of y_full_var for the filtered range
                y_filtered = y_full_var[(x1_full >= x_min) & (x1_full <= x_max)][mask1]
                return error_wrapper(y_filtered)

            error_jacobian = jacobian(full_error_wrapper)(y1_full)

            errors.append(float(error))
            jacobians[idx] = {
                'jacobian': error_jacobian,
                'common_column_values': x1_full.tolist()
            }

            comparison_details[idx] = {
                'error': float(error),
                'source_columns': source_col,
                'x_range': (float(x_min), float(x_max)),
                'filtered_indices': source_df_sorted.index[
                    (source_df_sorted[common_col] >= x_min) & (source_df_sorted[common_col] <= x_max)].tolist()
            }

        overall_error = np.nanmean(errors)

        dict_values = {
            'individual_errors': np.array(errors),
            'overall_error': overall_error,
            'comparison_details': comparison_details,
            'jacobians': jacobians
        }

        self.store_sos_outputs_values(dict_values)

    # def run(self):
    #     config_df = self.get_sosdisc_inputs("config_df")
    #     default_error_metric = self.get_sosdisc_inputs("default_error_metric")
    #     default_interpolation_method = self.get_sosdisc_inputs("default_interpolation_method")
    #     input_dict = self.get_sosdisc_inputs()
    #
    #     errors = []
    #     sources = []
    #     comparison_details = {}
    #     jacobians = {}
    #
    #     # Should not be running in case config_df is None
    #     if config_df is None:
    #         return
    #
    #     # perform comparison for each row
    #     for idx, row in config_df.iterrows():
    #         try:
    #             source_input_name = f"{row['variable_name']}"
    #             local_input_name = f"{row['local_var']}"
    #
    #             source_df = input_dict[source_input_name]
    #             local_df = input_dict[local_input_name]
    #
    #             # Get data from dataframes
    #             common_col = row["common_column"]
    #             source_col = row["column_name"]
    #             local_col = row["local_column"]
    #
    #             error_metric = row.get("error_metric", default_error_metric)
    #             interpolation_method = row.get("interpolation_method", default_interpolation_method)
    #
    #             # Sort data by the common column
    #             source_df_sorted = source_df.sort_values(by=common_col)
    #             local_df_sorted = local_df.sort_values(by=common_col)
    #
    #             # Find common x range
    #             x_min = max(
    #                 source_df_sorted[common_col].min(),
    #                 local_df_sorted[common_col].min(),
    #             )
    #             x_max = min(
    #                 source_df_sorted[common_col].max(),
    #                 local_df_sorted[common_col].max(),
    #             )
    #
    #             # Filter data within common range
    #             source_mask = (source_df_sorted[common_col] >= x_min) & (source_df_sorted[common_col] <= x_max)
    #             source_df_filtered = source_df_sorted[source_mask]
    #             local_df_filtered = local_df_sorted[
    #                 (local_df_sorted[common_col] >= x_min) & (local_df_sorted[common_col] <= x_max)]
    #
    #             x1 = anp.array(source_df_filtered[common_col].values)
    #             y1 = anp.array(source_df_filtered[source_col].values)
    #             x2 = anp.array(local_df_filtered[common_col].values)
    #             y_true = anp.array(local_df_filtered[local_col].values)
    #
    #             if interpolation_method.lower() != "none":
    #                 # Interpolation is on
    #                 def interpolation_wrapper(y1_var):
    #                     return self.interpolate_data_autograd(x1, y1_var, x2)
    #                 y_pred = interpolation_wrapper(y1)
    #             else:
    #                 # Interpolation is off, use only common x values
    #                 common_x = anp.intersect1d(x1, x2)
    #                 mask1 = anp.isin(x1, common_x)
    #                 mask2 = anp.isin(x2, common_x)
    #                 y1 = y1[mask1]
    #                 y_true = y_true[mask2]
    #                 y_pred = y1
    #
    #             def error_wrapper(y_var):
    #                 if interpolation_method.lower() != "none":
    #                     interpolated = interpolation_wrapper(y_var)
    #                 else:
    #                     interpolated = y_var
    #                 return self.compute_error_autograd(
    #                     y_true, interpolated, metric=error_metric
    #                 )
    #
    #             error = error_wrapper(y1)
    #             error_jacobian_filtered = jacobian(error_wrapper)(y1)
    #
    #             errors.append(float(error))
    #             sources.append(f"{source_input_name}:{source_col}")
    #
    #             # Create full-size Jacobian array with zeros for unused indices
    #             # full_jacobian = np.zeros(len(source_df_sorted))
    #             # full_jacobian[source_mask] = error_jacobian_filtered
    #
    #             jacobians[f"{source_input_name}:{source_col}"] = error_jacobian_filtered
    #
    #             comparison_details[idx] = {
    #                 "error": float(error),
    #                 "error_metric": error_metric,
    #                 "source_columns": source_col,
    #                 "x_range": (float(x_min), float(x_max)),
    #             }
    #
    #         except Exception as e:
    #             self.logger.error(f"Comparison failed for row {idx}: {e}")
    #             errors.append(np.nan)
    #             jacobians[idx] = None
    #
    #     overall_error = np.nanmean(errors)
    #
    #     dict_values = {
    #         "individual_errors": {k: v for k, v in zip(sources, errors)},
    #         "overall_error": overall_error,
    #         "comparison_details": comparison_details,
    #         "jacobians": jacobians,
    #     }
    #
    #     self.store_sos_outputs_values(dict_values)

    def compute_sos_jacobian(self):
        pass

    def get_chart_filter_list(self):
        chart_list = [
            "Comparisons"
        ]

        return [ChartFilter("Charts", chart_list, chart_list, "charts")]

    def get_post_processing_list(self, filters=None):
        comparison_config = self.get_sosdisc_inputs("config_df")
        instanciated_charts = []

        if filters is not None:
            for chart_filter in filters:
                if chart_filter.filter_key == 'charts':
                    charts = chart_filter.selected_values

        for _, row in comparison_config.iterrows():
            variable_name = row["variable_name"]
            variable_column = row["column_name"]
            input_var_name = row["local_var"]
            input_var_column = row["local_column"]
            common_column = row["common_column"]

            variable_data = self.get_sosdisc_inputs(variable_name)
            input_data = self.get_sosdisc_inputs(input_var_name)

            new_chart = TwoAxesInstanciatedChart(
                common_column,
                "Value",
                chart_name=f"Comparison of {input_var_name}:{input_var_column} vs {variable_name}:{variable_column}",
            )

            new_chart.add_series(
                InstanciatedSeries(
                    abscissa=variable_data[common_column],
                    ordinate=variable_data[variable_column],
                    series_name=f"{variable_name}: {variable_column}",
                    display_type="lines",
                    visible=True,
                )
            )

            new_chart.add_series(
                InstanciatedSeries(
                    abscissa=input_data[common_column],
                    ordinate=input_data[input_var_column],
                    series_name=f"{input_var_name}: {input_var_column}",
                    display_type="scatter",
                    visible=True,
                )
            )

            instanciated_charts.append(new_chart)

        return instanciated_charts

    @staticmethod
    def interpolate_data_autograd(x1, y1, x2, method="linear"):
        """
        Autograd-compatible interpolation

        Args:
            x1 (anp.ndarray): Original x values
            y1 (anp.ndarray): Original y values
            x2 (anp.ndarray): Target x values
            method (str): Interpolation method

        Returns:
            anp.ndarray: Interpolated values
        """

        # Linear interpolation implemented manually for autograd compatibility
        def linear_interpolate(x1, y1, x2):
            # Sort input arrays
            sort_idx = anp.argsort(x1)
            x1_sorted = x1[sort_idx]
            y1_sorted = y1[sort_idx]

            # Find insertion points
            idx = anp.searchsorted(x1_sorted, x2)

            # Handle edge cases
            idx = anp.clip(idx, 1, len(x1_sorted) - 1)

            # Linear interpolation
            x_left = x1_sorted[idx - 1]
            x_right = x1_sorted[idx]
            y_left = y1_sorted[idx - 1]
            y_right = y1_sorted[idx]

            # Interpolation weight
            w = (x2 - x_left) / (x_right - x_left)

            return y_left + w * (y_right - y_left)

        if method == "none":
            return y1
        elif method == "linear":
            return linear_interpolate(x1, y1, x2)

    @staticmethod
    def compute_error_autograd(y_true: np.ndarray, y_pred: np.ndarray, metric: str = "mse"):
        """
        Autograd-compatible error computation

        Args:
            y_true (anp.ndarray): True values
            y_pred (anp.ndarray): Predicted values
            metric (str): Error metric to compute

        Returns:
            float: Computed error
        """

        if metric == "mse":
            error = anp.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            error = anp.mean(anp.abs(y_true - y_pred))
        elif metric == "rmse":
            error = anp.sqrt(anp.mean((y_true - y_pred) ** 2))
        elif metric == "nrmse":
            error = anp.sqrt(anp.mean((y_true - y_pred) ** 2)) / anp.std(y_true)
        else:
            raise ValueError(f"Unsupported error metric: {metric}")

        return error

    @staticmethod
    def create_config_df(comparisons):
        """
        Create a configuration DataFrame for the AdvancedComparisonDisciplineWithJacobian.

        Args:
            comparisons (list): A list of dictionaries, each containing the configuration for one comparison.
                Each dictionary should have the following keys:
                - source_ns (str): Source namespace
                - variable_name (str): Variable name in the source namespace
                - column_name (str): Column name in the source dataframe
                - local_var (str): Local variable name
                - local_column (str): Column name in the local dataframe
                - common_column (str): Common column for comparison (e.g., 'time')
                - weight (float, optional): Weight for this comparison (default: 1.0)
                - interpolation_method (str, optional): Interpolation method (default: 'linear')

        Returns:
            pd.DataFrame: Configuration DataFrame ready to use in the discipline
        """
        required_keys = [
            "source_ns",
            "variable_name",
            "column_name",
            "local_var",
            "local_column",
            "common_column",
        ]
        optional_keys = {"weight": 1.0, "interpolation_method": "none", "error_metric": "mse"}

        config_data = []

        for comp in comparisons:
            # Check for required keys
            if not all(key in comp for key in required_keys):
                missing_keys = [key for key in required_keys if key not in comp]
                raise ValueError(
                    f"Missing required keys in comparison configuration: {missing_keys}"
                )

            # Add optional keys with default values if not provided
            for key, default_value in optional_keys.items():
                if key not in comp:
                    comp[key] = default_value

            config_data.append(comp)

        # Create DataFrame
        df = pd.DataFrame(config_data)

        # Ensure correct column order and data types
        column_order = [
            "source_ns",
            "variable_name",
            "column_name",
            "local_var",
            "local_column",
            "common_column",
            "error_metric",
            "weight",
            "interpolation_method",
        ]
        df = df[column_order]

        # Set data types
        df["source_ns"] = df["source_ns"].astype(str)
        df["variable_name"] = df["variable_name"].astype(str)
        df["column_name"] = df["column_name"].astype(str)
        df["local_var"] = df["local_var"].astype(str)
        df["local_column"] = df["local_column"].astype(str)
        df["common_column"] = df["common_column"].astype(str)
        df["weight"] = df["weight"].astype(float)
        df["interpolation_method"] = df["interpolation_method"].astype(str)
        df["error_metric"] = df["error_metric"].astype(str)

        return df
