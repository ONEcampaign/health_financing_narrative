import numpy as np
import pandas as pd
from bblocks.places import resolve_places
from oda_data import add_names_columns, set_data_path
from oda_data.indicators.research.sector_imputations import (
    imputed_multilateral_by_purpose,
    spending_by_purpose,
)
from oda_data.tools.sector_lists import add_broad_sectors

from scripts.config import Paths

set_data_path(Paths.raw_data)


def _rolling_value(df: pd.DataFrame) -> pd.DataFrame:
    recipients = df["recipient_code"].unique()
    years = np.arange(df["year"].min(), df["year"].max() + 1, dtype=np.int16)

    full_idx = pd.MultiIndex.from_product(
        [recipients, years], names=["recipient_code", "year"]
    )
    reindexed = (
        df.set_index(["recipient_code", "year"])["value"]
        .reindex(full_idx, fill_value=0.0)
        .sort_index()
    )

    # Rolling 3 years within each
    rolling = (
        reindexed.groupby(level=["recipient_code"])
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0], drop=True)
        .rename("value")
        .reset_index()
    )

    return rolling


def gov_health() -> pd.DataFrame:
    return (
        pd.read_csv(Paths.raw_data / "gov_health_spending.csv")
        .sort_values(by=["entity", "date"])
        .drop_duplicates(subset=["entity"], keep="last")
        .filter(["date", "entity", "entity_name", "variable", "unit", "value"])
        .assign(iso3_code=lambda d: d["entity"].str.split("/").str[1])
    )


def us_total_aid(year: int = 2023) -> pd.DataFrame:
    """Get total (bilateral + imputed multilateral) US aid to health."""

    bilateral = spending_by_purpose(
        years=2023,
        providers=302,
        measure="gross_disbursement",
        oda_only=True,
    )
    multilateral = imputed_multilateral_by_purpose(
        years=range(year - 4, year + 1),
        providers=302,
        measure="gross_disbursement",
    )

    total = (
        pd.concat([bilateral, multilateral])
        .pipe(add_broad_sectors)
        .loc[lambda d: d.purpose_name == "Health"]
        .assign(value=lambda d: d.value * 1e6)
        .groupby(["year", "recipient_code"], dropna=False)[["value"]]
        .sum()
        .reset_index()
        .pipe(_rolling_value)
        .loc[lambda d: d.year == year]
        .pipe(add_names_columns, "recipient_code")
    )

    total["iso3_code"] = resolve_places(
        total["recipient_name"], to_type="iso3_code", not_found="ignore"
    )

    return total.dropna(subset=["iso3_code"]).filter(["iso3_code", "value"])


def analysis() -> pd.DataFrame:
    """Produce data for a scatterplot comparing US aid to health vs government health spending."""

    return (
        gov_health()
        .merge(
            us_total_aid(2023), on="iso3_code", how="left", suffixes=("_gov", "_us_aid")
        )
        .assign(share=lambda d: round(100 * d["value_us_aid"] / d["value_gov"], 3))
        .loc[lambda d: d.share.notna()]
    )


if __name__ == "__main__":
    analysis().to_csv(Paths.output / "us_health_aid_vs_gov_health.csv", index=False)
