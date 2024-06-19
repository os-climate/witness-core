# IEA Data preparation Model


The model is designed to handle the interpolation and extrapolation of various energy-related datasets. The model processes multiple DataFrames representing different energy production, energy prices, CO2 taxes, population, and temperature, as provided by the International Energy Agency (IEA).
The model computes the "missing" data to get a yearly value
## Input Data

The input to the model consists of a dictionary of DataFrames. Each DataFrame represents a specific dataset provided by the IEA and includes:

- **Years**: A column representing the year for each data point.
- **Values**: Columns representing various metrics such as production quantities, prices, taxes, population, and temperature.

Example DataFrames:
- DataFrame for energy production of various technologies.
- DataFrame for energy prices.
- DataFrame for CO2 tax rates.
- DataFrame for population data.
- DataFrame for temperature data.

## Output Data

The output of the model is a dictionary of DataFrames similar to the input, but with the following modifications:

- The 'years' column spans from `year_start` to `year_end`.
- The values in the other columns are interpolated linearly for the years within the provided data range.
- If the data in any DataFrame starts after `year_start`, the model performs backward extrapolation to estimate values for the missing initial years.

## The Model

The model performs the following steps for each DataFrame in the input dictionary:

1. **Linear Interpolation**:
    - Apply linear interpolation to fill in the missing values within the range of available data points.

5. **Backward Extrapolation**:
    - Identify the first valid data point in each column.
    - Calculate the slope based on the first two valid data points.
    - Use this slope to perform linear backward extrapolation, estimating values for the years before the first valid data point.
