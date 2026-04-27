import logging
import re
from contextlib import contextmanager
from collections.abc import Generator

from hishel.httpx import SyncCacheClient
import polars as pl
from polars import selectors as cs
import fastexcel

LIFE_TABLE_TITLE_PATTERN = (
    r"^Table (\d+)\.? "
    + r"Life table for (?P<race>.*?)(?P<sex>male|female|)(s| population|)?: "
    + r"United States, (?P<year>\d{4})$"
)

# see https://www.cdc.gov/nchs/products/life_tables.htm
VOLUMES = {
    2001: "52_14",
    2002: "53_06",
    2003: "54_14",
    2004: "56_09",
    2005: "58_10",
    2006: "58_21",
    2007: "59_09",
    2008: "61_03",
    2009: "62_07",
    2010: "63_07",
    2011: "64_11",
    2012: "65_08",
    2013: "66_03",
    2014: "66_04",
    2015: "67_07",
    2016: "68_04",
    2017: "68_07",
    2018: "69-12",
    2019: "70-19",
    2020: "71-01",
    2021: "72-12",
    2022: "74-02",
    2023: "74-06",
}


# years 2001-2005 only published 9 tables; 2018 published 12
TABLE_COUNTS = {year: 9 for year in range(2001, 2006)} | {2018: 12}

BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR"
CLIENT = SyncCacheClient(base_url=BASE_URL)

LIFE_TABLE_COLS = ["q", "L", "e", "l", "d", "T"]

RACE_MAPPING = {
    # total
    "": "Pooled",
    "total": "Pooled",
    # hispanic
    "hispanic": "Hispanic",
    # white
    "white": "White",
    "white, non-hispanic": "Non-Hispanic White",
    "non-hispanic white": "Non-Hispanic White",
    # black
    "black": "Black",
    "black, non-hispanic": "Non-Hispanic Black",
    "non-hispanic black": "Non-Hispanic Black",
    # asian
    "asian, non-hispanic": "Non-Hispanic Asian",
    "non-hispanic asian": "Non-Hispanic Asian",
    # aian
    "american indian and alaska native, non-hispanic": "Non-Hispanic AIAN",
    "non-hispanic american indian or alaska native": "Non-Hispanic AIAN",
}


def get_all_life_tables() -> pl.DataFrame:
    """Fetch and combine all NCHS life tables across every available year and demographic group."""
    dfs = (
        get_life_table(year, table)
        for year in VOLUMES.keys()
        for table in range(1, TABLE_COUNTS.get(year, 18) + 1)
    )

    df = pl.concat(dfs, how="diagonal_relaxed").with_columns(
        pl.col("year").str.to_integer(),
        pl.col("sex").str.to_titlecase().replace({"": "Pooled"}).cast(pl.Categorical),
        pl.col("race")
        .str.to_lowercase()
        .str.strip_prefix("the ")
        .str.strip_suffix(" population")
        .str.replace("–", "-")
        .replace_strict(RACE_MAPPING, return_dtype=pl.Categorical),
    )

    assert df.drop_nulls().height == df.height

    sexes = df["sex"].unique().sort().to_list()
    assert sexes == ["Pooled", "Male", "Female"], f"found unexpected sex: {sexes}"
    assert df.filter(
        pl.col("table_title").str.to_lowercase().str.contains("male")
        & pl.col("sex").eq("Pooled")
    ).is_empty()

    assert df.select(pl.col("age").eq(0).any().over("table_title").all()).item(), (
        "not all tables have le at age zero"
    )

    assert df.select(pl.col("year").unique().len()).item() == len(VOLUMES), (
        "some years not accounted for"
    )

    assert df.filter(race="Pooled", sex="Pooled").unique("year").height == len(
        VOLUMES
    ), "pooled le not available in some years"

    return df


def get_life_table(year: int, table_number: int) -> pl.DataFrame:
    """Fetch a single NCHS life table by year and table number from the CDC FTP server."""
    logging.info(f"Fetching Table {table_number} for {year}")

    release = VOLUMES.get(year)
    assert release, f"no volume for {year}"

    stem = (
        f"Table{table_number:>02}_Intercensal"
        if year < 2010
        else f"Table{table_number:>02}"
    )

    # try both xlsx and xls extentions
    for ext in (".xlsx", ".xls"):
        response = CLIENT.get(f"{BASE_URL}/{release}/{stem}{ext}")

        if response.is_success:
            break

    response.raise_for_status()

    xl = fastexcel.read_excel(response.content)

    with _suppress_log_message("fastexcel.types.dtype", "falling back to string"):
        table_title = xl.load_sheet(0, n_rows=1).to_polars().columns[0].strip()
        table_contents = xl.load_sheet(
            0, header_row=2 if year > 2010 else 6
        ).to_polars()

    match = re.match(LIFE_TABLE_TITLE_PATTERN, table_title)
    assert match is not None, f"failed to parse {table_title}"
    assert match.group("year") == str(year), f"{match.group('year')} != {year}"

    return (
        table_contents.rename(lambda s: s.strip().replace("x", "").replace("()", ""))
        .select(
            pl.lit(table_title).alias("table_title"),
            *(pl.lit(v.strip()).alias(k) for k, v in match.groupdict().items()),
            (cs.starts_with("age") | cs.starts_with("Age") | cs.matches("__UNNAMED__0"))
            .str.extract(r"^(\d+).*$")
            .str.to_integer()
            .alias("age"),
            *LIFE_TABLE_COLS,
        )
        .drop_nulls("age")
    )


@contextmanager
def _suppress_log_message(logger_name: str, text: str) -> Generator[None]:
    """Suppress log records from `logger_name` whose message contains `text`."""
    logger = logging.getLogger(logger_name)
    f = logging.Filter()
    f.filter = lambda r: text not in r.getMessage()  # type: ignore[method-assign]
    logger.addFilter(f)

    try:
        yield

    finally:
        logger.removeFilter(f)


def main() -> None:
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    df = get_all_life_tables()

    logging.info(df)

    out = Path(__file__).parent / "nchs" / "life_tables.parquet"
    out.parent.mkdir(exist_ok=True)
    logging.info(f"saving to {out}")
    df.write_parquet(out)


if __name__ == "__main__":
    main()
