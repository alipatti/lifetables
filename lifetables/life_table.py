"""
See e.g. 
- https://www.ssa.gov/oact/HistEst/PerLifeTables/LifeTableDefinitions.pdf
- https://www.aitrs.org/sites/default/files/Life%20Tables.pdf
- https://mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf (p. 36)

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
from polars._typing import IntoExpr


def create_life_table(
    raw_mortality_rates: pl.LazyFrame,  # *by, age, raw_mortality_rate
    *,
    by: Iterable[str] = [],
    m: pl.Expr = pl.col("mortality"),
    age: pl.Expr = pl.col("age"),
    initial_cohort_size: IntoExpr = 1,
    separation_factor: IntoExpr = 0.5,
    infant_separation_factor: IntoExpr = 0.14,
    final_separation_factor: IntoExpr = 1 / pl.col("mortality"),
    q_equals_m=False,
) -> pl.LazyFrame:

    # TODO: handle age bins of width other than 1
    # width of age bin (next age - current age)
    # width = age.shift(-1) - age

    is_final_age = age == age.max()

    # mean number of years lived for those within bin (separation factor)
    s = (
        # infants
        pl.when(age=0)
        .then(infant_separation_factor)
        # TODO: handle final age bin
        .when(is_final_age)
        .then(final_separation_factor)
        # otherwise 1/2
        .otherwise(separation_factor)
    )

    # use q != m if populations are measured mid-year (after some number of people have died)
    q_inner = m if q_equals_m else m / (1 + (1 - s) * m)

    # probability of dying at age x, conditional on having lived until age x
    # top age bin gets set to 1
    q = pl.when(is_final_age).then(1.0).otherwise(q_inner)

    # probability of living to age x
    l = (1 - q).shift(1, fill_value=initial_cohort_size).cum_prod()

    # probability of dying at age x
    d = l * q

    # person years lived at exact age
    L = l - (1 - s) * d

    # person years remaining
    T = L.cum_sum(reverse=True)

    # life expectancy remaining
    e = T / l

    life_table_columns = dict(q=q, l=l, d=d, s=s, L=L, T=T, e=e)

    return raw_mortality_rates.sort(*by, "age").with_columns(
        expr.over(by).alias(name) if by else expr.alias(name)
        for name, expr in life_table_columns.items()
    )


def compute_le(
    mortality_rates: pl.LazyFrame,
    *,
    by: Iterable[str] = [],
    m: pl.Expr = pl.col("mortality"),
    age: pl.Expr = pl.col("age"),
    **kwargs,
) -> pl.LazyFrame:
    return create_life_table(mortality_rates, by=by, age=age, m=m, **kwargs).select(
        *by, age, le=pl.col("e") + age
    )
