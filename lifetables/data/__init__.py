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


def hmd_life_table(by_sex=True) -> pl.DataFrame:
    """
    From the Human Mortality Database:
    https://www.mortality.org/Country/Country?cntr=USA
    """
    if by_sex:
        return pl.concat(
            _hmd_life_table(sex).select(
                pl.lit(sex).cast(SEX_TYPE).alias("sex"), pl.all()
            )
            for sex in ("Male", "Female")
        )

    return _hmd_life_table("Both")
