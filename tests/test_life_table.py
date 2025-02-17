import polars as pl

from lifetables.life_table import create_life_table
from lifetables.data import hmd_life_table


def test_life_table() -> None:
    """
    Test to check whether we match the life expectancy estimates
    of the Human Mortality Database when using their mortality rates.
    """

    # original life table from HMD
    original = hmd_life_table().sort(pl.all())

    # recreation using our life table function
    recreation = (
        original.lazy()
        .select("year", "age", "sex", "m", "a")
        .pipe(
            create_life_table,
            by=["year", "sex"],
            m=pl.col("m"),
            final_separation_factor=1 / pl.col("m"),
            infant_separation_factor=pl.col("a"),
            initial_cohort_size=100_000,
        )
        .sort(pl.all())
        .drop("a")
        .collect()
    )

    # compare LE estimates
    error = pl.col("e").sub("e_right")
    mse, mean, max = (
        original.join(recreation, on=["year", "sex", "age"])
        .select(
            mse=error.pow(2).mean(),
            mean=error.mean(),
            max=error.abs().max(),
        )
        .iter_rows()
        .__next__()
    )

    assert max < 0.01
    assert mse < 1e-5
    assert abs(mean) < 1e-5
