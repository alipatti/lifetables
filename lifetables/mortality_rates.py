from typing import Iterable

import polars as pl
from polars import selectors as cs


def smooth_mortality_rates():
    raise NotImplementedError


def fill_mortality_rates(
    original_rates: pl.LazyFrame,
    rates_to_fill_with: pl.LazyFrame,
    *,
    by: Iterable[str],  # e.g. ["year", "sex", "race", "cause_of_death"]
    join_on: Iterable[str],  # e.g. ["year", "sex"] or ["year"]
    min_age=30,
    switch_age=85,
    max_age=110,
    pre_switch_fill_value=float("nan"),
    post_switch_fill_value=float("nan"),
    column_to_fill="mortality",
) -> pl.LazyFrame:

    # construct spine containing the full age range of interest
    ages = pl.int_range(min_age, max_age + 1, step=1, eager=True).to_frame("age").lazy()
    spine = original_rates.select(*by).unique().sort(pl.all()).join(ages, how="cross")

    # populate spine with both original rates and fill rates
    both_rates = spine.join(
        original_rates,
        how="left",
        on=list(by) + ["age"],
        validate="1:1",
    ).join(
        rates_to_fill_with.select(*join_on, "age", column_to_fill),
        how="left",
        on=list(join_on) + ["age"],
        validate="m:1",
        suffix="_fill",
    )

    return (
        both_rates.with_columns(
            # before swich_age, use original rates
            pl.when(pl.col("age").lt(switch_age)).then(
                pl.col(column_to_fill).fill_null(value=pre_switch_fill_value)
            )
            # on and after swich_age, use new rates
            .otherwise(
                pl.col(column_to_fill + "_fill").fill_null(value=post_switch_fill_value)
            )
        )
        # drop the fill rates
        .drop(cs.ends_with("_fill")).sort(*by, "age")
    )
