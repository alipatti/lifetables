from typing import Optional, Literal
from importlib import resources

import polars as pl
from polars import selectors as cs


DATA = resources.files("lifetables.data")
SEX_TYPE = pl.Enum(("Male", "Female"))


def _hmd_life_table(
    sex: Literal["Male", "Female", "Both"],
) -> pl.DataFrame:

    path = DATA / "hmd" / f"{sex[0].lower()}ltper_1x1.txt"

    lines = [l.split() for l in path.read_text().splitlines()[2:]]

    return pl.from_records(lines[1:], orient="row", schema=lines[0]).select(
        pl.col("Year", "Age")
        .str.extract(r"(\d+)")
        .str.to_integer()
        .name.to_lowercase(),
        cs.ends_with("x").cast(pl.Float64).name.map(lambda s: s.strip("x")),
    )


def nchs_life_table() -> pl.DataFrame:
    """
    Get U.S. life tables from the National Center for Health Statistics.

    Data are aggregated from individual Excel files published annually in the
    National Vital Statistics Reports series and hosted on the CDC FTP server
    at https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR.

    For a list of all annual reports, see https://www.cdc.gov/nchs/products/life_tables.htm.
    """
    path = DATA / "nchs" / "life_tables.parquet"
    return pl.read_parquet(path.open("rb"))


def hmd_life_table(by_sex=True) -> pl.DataFrame:
    """
    Get U.S. life tables from the Human Mortality Database.

    See https://www.mortality.org/Country/Country?cntr=USA for more.
    """
    if by_sex:
        return pl.concat(
            _hmd_life_table(sex).select(
                pl.lit(sex).cast(SEX_TYPE).alias("sex"), pl.all()
            )
            for sex in ("Male", "Female")
        )

    return _hmd_life_table("Both")
