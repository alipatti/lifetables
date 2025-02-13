"""
original paper: https://doi.org/10.2307/2061029
more concrete implementation: https://doi.org/10.1016/j.annepidem.2014.05.006
"""

from typing import Iterable, Literal, Optional

import polars as pl
from polars import selectors as cs

from lifetables.helpers import aggregate_cod
from lifetables.life_table import create_life_table

_SHARE_WITHIN_AGE = "share_within_age"


def _arriaga_decomposition_by_age(
    initial_mortality_rates: pl.LazyFrame,
    new_mortality_rates: pl.LazyFrame,
    *,
    by: Iterable[str] = [],
):

    q, l, L = pl.col("mortality"), pl.col("l"), pl.col("L")
    q_new, l_new, L_new = pl.col("mortality_new"), pl.col("l_new"), pl.col("L_new")

    e_new = pl.col("e_new")
    T_new = pl.col("T_new")

    direct_effect = (l * (L_new / l_new - L / l)).alias("direct_effect")
    indirect_effect = (T_new.shift(-1) * (l / l_new).diff(-1)).alias("indirect_effect")

    # gap contributions
    years = (direct_effect + indirect_effect).alias("contribution_years")
    proportion = (years / years.sum()).alias("contribution_proportion")

    return (
        initial_mortality_rates.pipe(create_life_table, by=by)
        .join(
            new_mortality_rates.pipe(create_life_table, by=by),
            on=list(by) + ["age"],
            how="inner",
            validate="1:1",
            suffix="_new",
        )
        .select(
            *by,
            "age",
            # direct_effect.over(by),
            # indirect_effect.over(by),
            years.over(by),
            proportion.over(by),
        )
    )


def _arriaga_decomposition_by_cause_within_age(
    initial_mortality_rates: pl.LazyFrame,
    new_mortality_rates: pl.LazyFrame,
    *,
    by: Iterable[str],
    cause_column="cause_of_death",
):
    """
    returns df w/ columns `*by, age, cause_column, share_within_age`
    """

    q = pl.col("mortality")
    q_new = pl.col("mortality_new")
    cause_share = (q_new - q) / (q_new - q).sum().over(*by, "age")

    return initial_mortality_rates.join(
        new_mortality_rates,
        on=list(by) + ["age", cause_column],
        how="inner",
        validate="1:1",
        suffix="_new",
    ).select(*by, "age", cause_column, cause_share.alias(_SHARE_WITHIN_AGE))


def _arriaga_decomposition(
    initial_mortality_rates: pl.LazyFrame,
    new_mortality_rates: pl.LazyFrame,
    *,
    by: Iterable[str] = [],
    within_age: Optional[str] = None,  # e.g. cause_of_death
):
    """
    returns df w/ columns `*by, age, cause_column?, contribution_years, contribution_proportion`
    """

    # calculate contributions to gap by age
    contributions_by_age = _arriaga_decomposition_by_age(
        initial_mortality_rates.pipe(aggregate_cod, by=by),
        new_mortality_rates.pipe(aggregate_cod, by=by),
        by=by,
    )

    if not within_age:
        return contributions_by_age

    # calculate contribution shares within age
    shares_within_age = _arriaga_decomposition_by_cause_within_age(
        initial_mortality_rates,
        new_mortality_rates,
        by=by,
        cause_column=within_age,
    )

    # multiply total age contribution by within-age share
    # to get total contribution by cause x age
    return (
        shares_within_age.join(
            contributions_by_age,
            on=list(by) + ["age"],
            how="inner",
            validate="m:1",
        )
        .with_columns(cs.starts_with("contribution_") * pl.col(_SHARE_WITHIN_AGE))
        .drop(_SHARE_WITHIN_AGE)
    )


def arriaga_decomposition(
    initial_mortality: pl.LazyFrame,
    new_mortality: Optional[pl.LazyFrame] = None,
    *,
    by: Iterable[str] = [],
    direction: Literal["forward", "backward", "average"] = "average",
    within_age: Optional[str] = None,  # e.g. cause_of_death
) -> pl.LazyFrame:

    # if other mortality rates are not passed, set them uniformly to zero
    # to decompose the le toll by age/cause
    if new_mortality is None:
        new_mortality = initial_mortality.with_columns(pl.lit(0).alias("mortality"))

    if direction == "forward":
        return _arriaga_decomposition(
            initial_mortality,
            new_mortality,
            by=by,
            within_age=within_age,
        )

    if direction == "backward":
        return _arriaga_decomposition(
            new_mortality,  # flip initial and new
            initial_mortality,
            by=by,
            within_age=within_age,
        ).with_columns(pl.col("contribution_years").mul(-1)) # correct negative sign in years

    # now direction == "average":

    directions = ["forward", "backward"]

    both_directions = pl.concat(
        arriaga_decomposition(
            initial_mortality,
            new_mortality,
            by=by,
            direction=d,  # type: ignore
            within_age=within_age,
        ).with_columns(pl.lit(d).alias("direction"))
        for d in directions
    )

    group = [*by, "age", within_age] if within_age else [*by, "age"]
    return both_directions.group_by(group, maintain_order=True).agg(
        cs.exclude("direction").mean()
    )
