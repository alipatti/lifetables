from typing import Iterable

import polars as pl


def aggregate_cod(
    cause_specific_rates: pl.LazyFrame,
    by: Iterable[str],
    other_exprs: Iterable[pl.Expr] = [],
) -> pl.LazyFrame:
    """
    Aggregates age-specific COD across causes of death and
    cancer sites within a particular cell.
    """
    return cause_specific_rates.group_by(*by, "age", maintain_order=True).agg(
        pl.col("mortality").sum(), *other_exprs
    )
