from typing import Iterable

import polars as pl

from lifetables.life_table import create_life_table


def arriaga_decomposition_by_age(
    initial_mortality_rates: pl.LazyFrame,
    new_mortality_rates: pl.LazyFrame,
    *,
    by: Iterable[str] = [],
):

    q, l, L = pl.col("mortality"), pl.col("l"), pl.col("L")
    q_new, l_new, L_new = pl.col("mortality_new"), pl.col("l_new"), pl.col("L_new")

    e_new = pl.col("e_new")
    T_new = pl.col("T_new")

    # old, more complicated method
    direct_effect = (l * (L_new / l_new - L / l)).alias("direct_effect")
    indirect_effect = (T_new.shift(-1) * (l / l_new).diff(-1)).alias("indirect_effect")

    # simplified form derived by Hannes Schwandt (and verified by Alistair)
    # direct_effect = (l * (q - q_new) * pl.col("s_new")).alias("direct_effect")
    # indirect_effect = (l * (q - q_new) * e_new.shift(-1)).alias("indirect_effect")

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
            direct_effect.over(by),
            indirect_effect.over(by),
            years.over(by),
            proportion.over(by),
        )
    )
