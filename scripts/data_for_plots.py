import pandas as pd

from scripts.config import Paths
from scripts.common import (
    get_ghed,
    add_country_info,
    merge_ghed,
)

from scripts.logger import logger


def heatmap_available_indicators(ghed_df: pd.DataFrame) -> None:

    available_pct = (
        ghed_df.groupby(["year", "iso3_code"])["value"]
        .apply(lambda s: 100 - s.isna().mean() * 100)
        .reset_index(name="available_pct")
    )

    years = pd.Index(available_pct["year"].unique(), name="year")
    countries = pd.Index(available_pct["iso3_code"].unique(), name="iso3_code")

    full_idx = pd.MultiIndex.from_product(
        [years, countries], names=["year", "iso3_code"]
    )
    available_pct_full = (
        available_pct.set_index(["year", "iso3_code"])
        .reindex(full_idx)
        .reset_index()
        .fillna(0)
    )

    available_pct_full = add_country_info(available_pct_full).sort_values(
        ["country_name", "year"]
    )

    available_pct_full.to_csv(
        Paths.output / "chart_data_available_indicators.csv", index=False
    )
    logger.info("Exported available indicators chart and data")


def scatter_gghe_pc_vs_life_expectancy(ghed_df: pd.DataFrame) -> None:

    le_indicators = ["WHOSIS_000001"]  # Life expectancy at birth
    le_sexes = ["SEX_BTSX"]  # both sexes
    le_columns = {
        "SpatialDimValueCode": "iso3_code",
        "Period": "year",
        "Value": "Life expectancy (years)",
    }
    le_df_raw = pd.read_csv(Paths.life_expectancy)
    le_df_raw["Value"] = le_df_raw["Value"].str.extract(r"^([\d\.]+)").astype(float)
    le_df = le_df_raw.query(
        "IndicatorCode in @le_indicators and Dim1ValueCode in @le_sexes"
    )[le_columns.keys()].rename(columns=le_columns)

    ghed_indicators = ["gghed_usd2022_pc"]

    df = merge_ghed(ghed_df, ghed_indicators, le_df, ["Life expectancy (years)"])

    df_download = df.drop(columns=["iso3_code", "region"]).rename(
        columns={
            ghed_indicators[0]: "Government health expenditure per capita",
            "income_level": "Income level",
            "country_name": "Country",
        }
    )

    # export chart and data
    df_download.to_csv(
        Paths.output / "chart_data_government_spending_per_capita_life_expectancy.csv",
        index=False,
    )
    df.to_csv(
        Paths.output / "chart_government_spending_per_capita_life_expectancy.csv",
        index=False,
    )
    logger.info(
        "Exported government spending per capita vs. life expectancy chart and data"
    )


def _add_roll_avgs(
    df: pd.DataFrame, roll_periods: int = 5, min_periods: int = 5
) -> pd.DataFrame:
    """Add rolling average column to a dataframe with columns country_name, indicator_code, year, value

    Args:
        df: Dataframe with columns country_name, indicator_code, year, value
        roll_periods: Number of periods to use for rolling average. Defaults to 5.
        min_periods: Minimum number of periods to use for rolling average. Defaults to 5.

    """

    return df.assign(
        rolling_avg=lambda d: d.groupby(["country_name", "indicator_code"])[
            "value"
        ].transform(lambda x: x.rolling(roll_periods, min_periods=min_periods).mean())
    )


def chart_bar_immunisation(ghed_df: pd.DataFrame) -> None:
    """Create chart data for immunisation external share chart"""

    immunisation_inds = [
        "hc62_ext_usd2022",  # external spending for immunisation programs
        "hc62_usd2022",  # total spending for immunisation programs
    ]
    # preprocess the data
    df = (
        ghed_df.loc[
            lambda d: d.indicator_code.isin(immunisation_inds),
            ["country_name", "year", "indicator_code", "value"],
        ]
        .dropna(subset=["value"])
        .loc[
            lambda d: d.groupby(["country_name", "indicator_code"])["year"].transform(
                "max"
            )
            >= 2021
        ]  # keep only countries with data in 2021 or later
        .reset_index(drop=True)
    )

    # add rolling averages
    df = _add_roll_avgs(df, roll_periods=5, min_periods=3)

    # calculate external share of total
    df = (
        df.pivot(
            index=["country_name", "year"],
            columns="indicator_code",
            values="rolling_avg",
        )
        .reset_index()
        .assign(ext_pct=lambda d: d.hc62_ext_usd2022 / d.hc62_usd2022 * 100)
    )

    # filter the data for countries with at least 50% external share in most recent year, where recent year is 2021 or later
    df = (
        df.loc[
            lambda d: d.groupby("country_name")["year"].transform("max") == d["year"]
        ]
        .loc[lambda d: d.ext_pct >= 50]
        .sort_values("ext_pct", ascending=False)
    )

    # format data for downloadable csv
    df_download = df.rename(
        columns={
            "hc62_ext_usd2022": "External spending on immunisation programs (5yr rolling avg, millions USD constant 2022)",
            "hc62_usd2022": "Total spending on immunisation programs (5yr rolling avg, millions USD constant 2022)",
            "ext_pct": "External spending as % of total immunisation spending (5yr rolling avg)",
        }
    )

    # export chart and data
    df_download.to_csv(
        Paths.output / "chart_data_immunisation_external_share.csv", index=False
    )
    df.to_csv(Paths.output / "chart_immunisation_external_share.csv", index=False)
    logger.info("Exported immunisation external share chart and data")


if __name__ == "__main__":
    # GHED_DF = get_ghed()

    scatter_gghe_pc_vs_life_expectancy(GHED_DF)

    # heatmap_available_indicators(GHED_DF)
    # chart_bar_immunisation(GHED_DF)
