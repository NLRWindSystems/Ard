import pandas as pd
import yaml
from pathlib import Path
import numpy as np
import csv
import datetime as dt

import jsonschema.exceptions
import pytest

import windIO


def load_header_key_value(csv_path: str) -> dict:
    """
    Read first two lines of a CSV. First line -> keys, second line -> values.
    Returns a dict mapping key -> value (strings converted to float/int where possible).
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV has no lines.")
        try:
            values = next(reader)
        except StopIteration:
            raise ValueError("CSV has only one line; need two.")
    if len(values) != len(header):
        raise ValueError("Header and value line have different lengths.")

    def _convert(v: str):
        v = v.strip()
        if v == "":
            return v
        for cast in (int, float):
            try:
                return cast(v)
            except Exception:
                pass
        return v

    return {k.strip(): _convert(v) for k, v in zip(header, values)}


def build_dt(datetime_str, tz_str):

    # tz_str like 'GMT+2' or 'GMT-5'
    if not tz_str.startswith("GMT") or len(tz_str) < 5:
        raise ValueError("Expected format 'GMT+H' or 'GMT-H'")
    sign = tz_str[3]
    offset_hours = int(tz_str[4:])
    if sign == "-":
        offset_hours = -offset_hours
    tz = dt.timezone(dt.timedelta(hours=offset_hours))

    datetime_obj_naive = dt.datetime.fromisoformat(datetime_str)

    datetime_obj_aware = dt.datetime(
        year=datetime_obj_naive.year,
        month=datetime_obj_naive.month,
        day=datetime_obj_naive.day,
        hour=datetime_obj_naive.hour,
        second=datetime_obj_naive.second,
        tzinfo=tz,
    )

    return datetime_obj_naive.isoformat() + "Z"


def extract_columns_to_yaml(
    input_csv: str, output_yaml: str, columns: list, turbulence_intensity: float
):
    """
    Load a CSV file, extract specific columns, and save them to a YAML file in the desired format.

    Args:
        input_csv (str): Path to the input CSV file.
        output_yaml (str): Path to the output YAML file.
        columns (list): List of column names to extract.
        turbulence_intensity (float): Constant turbulence intensity to include in the output.

    Returns:
        None
    """

    # time = datetime(2020, 10, 31, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
    meta_data = load_header_key_value(input_csv)

    # Load the CSV file
    try:
        data = pd.read_csv(input_csv, header=2)
    except FileNotFoundError:
        print(f"Error: The file '{input_csv}' was not found.")
        return
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return

    # Check if all desired columns exist in the file
    missing_columns = [col for col in columns if col not in data.columns]
    if missing_columns:
        print(
            f"Error: The following columns are missing in the input file: {missing_columns}"
        )
        return

    # Extract the desired columns
    extracted_data = data[columns]  # .head(10)

    # Prepare the YAML data structure
    yaml_data = {
        "name": input_csv,
        "wind_resource": {
            "time": [
                build_dt(t, tz_str=meta_data["timezone_abbreviation"])
                for t in extracted_data[columns[0]]
            ],
            "wind_speed": extracted_data[columns[1]].tolist(),
            "wind_direction": extracted_data[columns[2]].tolist(),
            "turbulence_intensity": {
                "data": [turbulence_intensity] * len(extracted_data),
            },
        },
    }

    # Save the data to a YAML file
    try:
        with open(output_yaml, "w") as yaml_file:
            yaml.dump(yaml_data, yaml_file, default_flow_style=False)
        print(f"Extracted data saved to '{output_yaml}'")
    except Exception as e:
        print(f"Error: An error occurred while saving the file: {e}")


# Example usage
if __name__ == "__main__":
    input_file = "open-meteo-56.20N8.54E86m.csv"  # Replace with your input file path
    output_file = (
        "open-meteo-56.20N8.54E86m.yaml"  # Replace with your desired output file path
    )
    desired_columns = ["time", "wind_speed_100m (m/s)", "wind_direction_100m (°)"]

    extract_columns_to_yaml(
        input_file, output_file, desired_columns, turbulence_intensity=0.1
    )

    if windIO.validate(
        input=output_file,
        schema_type="plant/energy_resource",
    ):
        print("output validated")
