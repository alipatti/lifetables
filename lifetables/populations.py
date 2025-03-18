import pandas as pd
import polars as pl


import polars.selectors as cs


def get_standard_pops(
    url="https://seer.cancer.gov/stdpopulations/stdpop.singleages.html",
    age_column=cs.by_index(0),
    pop_column=cs.by_index(2), # single ages to 84
):
    raw = pl.from_pandas(pd.read_html(url).pop())

    return raw.select(
        age_column.str.extract(r"(\d{2}) years")
        .str.to_integer(strict=False)
        .alias("age"),
        pop_column.alias("population"),
    ).drop_nulls()
