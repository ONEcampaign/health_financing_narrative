import pandas as pd

from scripts.config import Paths
from common import (
    get_ghed,
    add_country_info,
    merge_ghed,
)

def get_available_indicators_by_country_year(
    ghed_df: pd.DataFrame, save_csv: bool = True
) -> pd.DataFrame:

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

    if save_csv:
        available_pct_full.to_csv(Paths.output / "available_pct.csv", index=False)

    return available_pct_full

def get_che_life_expectancy(ghed_df: pd.DataFrame) -> pd.DataFrame:

    le_indicators = ["WHOSIS_000001"]  # Life expectancy at birth
    le_sexes = ["SEX_BTSX"]  # both sexes
    le_columns = {
        "SpatialDimValueCode": "iso3_code",
        "Period": "year",
        "Value": "le_birth",
    }
    le_df_raw = pd.read_csv(Paths.life_expectancy)
    le_df_raw["Value"] = le_df_raw["Value"].str.extract(r"^([\d\.]+)").astype(float)
    le_df = le_df_raw.query(
        "IndicatorCode in @le_indicators and Dim1ValueCode in @le_sexes"
    )[le_columns.keys()].rename(columns=le_columns)

    ghed_indicators = ["che_gdp", "che_pc_usd"]

    df_merged = merge_ghed(ghed_df, ghed_indicators, le_df, ["le_birth"], "che_le")

    return df_merged

if __name__ == "__main__":
    ghed_df = get_ghed()
    available_pct_full = get_available_indicators_by_country_year(ghed_df)
    che_le = get_che_life_expectancy(ghed_df)
