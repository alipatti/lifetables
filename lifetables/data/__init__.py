from typing import Optional, Literal
from importlib import resources

import polars as pl
from polars import selectors as cs


DATA = resources.files("lifetables.data")


def hmd_life_table(
    sex: Optional[Literal["Male", "Female", "Pooled"]] = None,
) -> pl.DataFrame:
    """
    From the Human Mortality Database:
    https://www.mortality.org/Country/Country?cntr=USA
    """
    if not sex:
        return pl.concat(hmd_life_table(sex) for sex in ("Male", "Female", "Pooled"))

    path = DATA / "hmd" / f"{('b' if sex == 'Pooled' else sex[0]).lower()}ltper_1x1.txt"

    lines = [l.split() for l in path.read_text().splitlines()[2:]]

    return pl.from_records(lines[1:], orient="row", schema=lines[0]).select(
        pl.col("Year", "Age")
        .str.extract(r"(\d+)")
        .str.to_integer()
        .name.to_lowercase(),
        pl.lit(sex).alias("sex"),
        cs.ends_with("x").cast(pl.Float64).name.map(lambda s: s.strip("x")),
    )
