"""
Life table functions as defined at 
https://www.ssa.gov/oact/HistEst/PerLifeTables/LifeTableDefinitions.pdf.

Expects columns "mortality" and "age", with the data frame sorted (increasing)
by age, for example
```python
┌─────┬───────────┐
│ age ┆ mortality │
│ --- ┆ ---       │
│ i64 ┆ f64       │
╞═════╪═══════════╡
│ 30  ┆ 0.00099   │
│ 31  ┆ 0.001018  │
│ 32  ┆ 0.001047  │
│ 33  ┆ 0.001076  │
│ 34  ┆ 0.001104  │
│ …   ┆ …         │
│ 80  ┆ 0.000004  │
│ 81  ┆ 0.000004  │
│ 82  ┆ 0.000004  │
│ 83  ┆ 0.000005  │
│ 84  ┆ 0.000005  │
└─────┴───────────┘
```
"""

from typing import Iterable
import polars as pl


def create_life_table(
    df: pl.LazyFrame,
    *,
    by: Iterable[str] = [],
    raw_mortality_rate=pl.col("mortality"),
    age=pl.col("age"),
    l_0=1,
) -> pl.LazyFrame:

    # TODO: account for variable-width bins?
    # width of age bin (next age - current age)
    # n = age.shift(-1) - age

    # probability of dying at age x, conditional on having lived until age x
    q = pl.when(age == age.max()).then(1.0).otherwise(raw_mortality_rate)
    p = 1 - q

    # probability of living to age x
    l = p.shift(1, fill_value=l_0).cum_prod()

    # probability of dying at age x
    d = l * q

    # mean number of years lived for those within bin
    s = pl.when(age == 0).then(0.14).otherwise(0.5)

    # person years lived at exact age
    L = l - s * d

    # person years remaining
    T = L.cum_sum(reverse=True)

    # life expectancy remaining
    e = T / l

    le = e + pl.col("age")

    life_table_columns = [q, l, d, L, T, e]

    return df.sort(*by, "age").with_columns(
        col.over(by) if by else col for col in life_table_columns
    )
