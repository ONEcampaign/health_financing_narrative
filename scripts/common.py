import pandas as pd

from bblocks.data_importers import GHED
from bblocks.places import resolve_places

from scripts.config import Paths


def get_ghed() -> pd.DataFrame:
    ghed = GHED()
    data = ghed.get_data()
    return data


def filter_ghed_indicators(
    ghed_df: pd.DataFrame, indicators: list[str]
) -> pd.DataFrame:

    cols = ["iso3_code", "year", "indicator_code", "value"]
    df_filtered = ghed_df.query("indicator_code in @indicators")
    df = df_filtered.pivot(
        index=["iso3_code", "year"], columns="indicator_code", values="value"
    ).reset_index()

    return df


def add_country_info(
    df: pd.DataFrame, country_column: str = "iso3_code"
) -> pd.DataFrame:

    if country_column != "iso3_code":
        df["iso3_code"] = resolve_places(
            df[country_column], to_type="iso3_code", not_found="ignore"
        )

    df["country_name"] = resolve_places(
        df["iso3_code"], to_type="name_short", not_found="ignore"
    )
    df.dropna(subset=["country_name"], inplace=True)
    df["region"] = resolve_places(df["iso3_code"], to_type="region", not_found="ignore")
    df["income_level"] = resolve_places(
        df["iso3_code"], to_type="income_level", not_found="ignore"
    )

    missing_map = {
        "NIU": "Upper middle income",
        "VEN": "Upper middle income",
        "COK": "High income",
    }

    df["income_level"] = df["iso3_code"].map(missing_map).fillna(df["income_level"])

    return df


def merge_ghed(
    ghed_df: pd.DataFrame,
    ghed_indicators: list,
    df2: pd.DataFrame,
    df2_indicators: list
) -> pd.DataFrame:

    ghed_df = filter_ghed_indicators(ghed_df, ghed_indicators)

    full_df = pd.merge(ghed_df, df2, how="outer", on=["iso3_code", "year"])

    full_df = add_country_info(full_df)

    all_indicators = ghed_indicators + df2_indicators

    avg_df = avg_with_complete_years(full_df, all_indicators)

    return avg_df


def split_column(
    df: pd.DataFrame,
    col: str,
    new_col1: str,
    new_col2: str,
    sep: str = ":",
    remove_original: bool = True,
) -> pd.DataFrame:
    """
    Split a column by a separator into two new columns.
    """
    df[[new_col1, new_col2]] = df[col].str.split(sep, n=1, expand=True)

    if remove_original:
        df = df.drop(columns=[col])

    return df


def avg_with_complete_years(
    df: pd.DataFrame,
    value_cols: list[str],
    start_year: int = 2010,
    end_year: int = 2021,
    group_vars: list[str] = ["country_name", "iso3_code", "region", "income_level"],
) -> pd.DataFrame:
    """
    Calculate averages of given value columns for countries
    with complete data in the given year range.
    """
    # Filter to year range
    df_filtered = df[df["year"].between(start_year, end_year)]

    # Identify complete countries
    required_years = set(range(start_year, end_year + 1))
    complete_countries = (
        df_filtered.dropna(subset=value_cols)
        .groupby("country_name")["year"]
        .nunique()
        .loc[lambda s: s == len(required_years)]
        .index
    )

    # Keep only complete countries
    df_complete = df_filtered[df_filtered["country_name"].isin(complete_countries)]

    # Average across years
    result = df_complete.groupby(group_vars)[value_cols].mean().reset_index()

    result["year"] = str(start_year) + "-" + str(end_year)

    return result


def format_who_df(
    df: pd.DataFrame, value_col: str = "mortality", disease: None | str = None
) -> pd.DataFrame:

    column_map = {
        "SpatialDimValueCode": "iso3_code",
        "Period": "year",
        "FactValueNumeric": value_col,
    }

    df = df[column_map.keys()].rename(columns=column_map)

    if disease is not None:
        df["disease"] = disease

    return df


def compute_pct(
    df: pd.DataFrame, abs_cols: list[str], total_col: str, suffix: str = "pct"
) -> pd.DataFrame:
    df = df.copy()
    for c in abs_cols:
        df[f"{c}_{suffix}"] = df[c] / df[total_col] * 100
    return df


def add_population(df: pd.DataFrame) -> pd.DataFrame:

    cols_map = {"Time": "year", "Iso3": "iso3_code", "Value": "population"}

    pop_raw = pd.read_csv(Paths.population)
    pop = pop_raw[cols_map.keys()].rename(columns=cols_map)

    merged_df = pd.merge(df, pop, on=["iso3_code", "year"], how="left")

    return merged_df


def compute_per_capita(
    df: pd.DataFrame,
    cols: list[str],
    pop_col: str = "population",
    suffix: str = "pc",
    factor: int = 1_000_000,
) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[f"{c}_{suffix}"] = df[c] * factor / df[pop_col]
    return df
