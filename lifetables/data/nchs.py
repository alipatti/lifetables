import logging
import re

from hishel.httpx import SyncCacheClient
import polars as pl
import fastexcel

LIFE_TABLE_TITLE_PATTERN = (
    r"^Table (?P<table_number>\d+)\.? "
    + r"Life table for (?P<race>.*?)(?P<sex>males|females|): "
    + r"United States, (?P<year>\d{4})$"
)

# see https://www.cdc.gov/nchs/products/life_tables.htm
VOLUMES = {
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


BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR"
CLIENT = SyncCacheClient(base_url=BASE_URL)

LIFE_TABLE_COLS = ["q", "L", "e", "l", "d", "T"]

RACE_MAPPING = {
    # total
    "": "Total",
    "total": "Total",
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
    dfs = (
        get_life_table(year, table)
        for year in VOLUMES.keys()
        for table in (range(1, 19) if year != 2018 else range(1, 13))
    )

    df = pl.concat(dfs, how="diagonal_relaxed").with_columns(
        pl.col("year").str.to_integer(),
        pl.col("sex").replace_strict(
            {"females": "Female", "males": "Male", "": "Pooled"},
            return_dtype=pl.Categorical,
        ),
        pl.col("race")
        .str.to_lowercase()
        .str.strip_prefix("the ")
        .str.strip_suffix(" population")
        .str.replace("–", "-")
        .replace_strict(RACE_MAPPING, return_dtype=pl.Categorical),
    )

    assert df.drop_nulls().height == df.height

    assert df.select(pl.col("age").eq(0).any().over("year").all()).item(), (
        "not all years have le at age zero"
    )

    assert df.select(pl.col("year").unique().len()).item() == len(VOLUMES)

    return df


def get_life_table(year: int, table_number: int) -> pl.DataFrame:

    logging.info(f"Fetching Table {table_number} for {year}")

    release = VOLUMES.get(year)
    assert release, f"no volume for {year}"

    filename = (
        f"Table{table_number:>02}_Intercensal.xlsx"
        if year < 2010
        else f"Table{table_number:>02}.xlsx"
    )
    url = f"{BASE_URL}/{release}/{filename}"
    response = CLIENT.get(url)
    response.raise_for_status()

    xl = fastexcel.read_excel(response.content)

    # get the table title from cell AA
    table_title = xl.load_sheet(0, n_rows=1).to_polars().columns[0].strip()

    match = re.match(LIFE_TABLE_TITLE_PATTERN, table_title)
    assert match is not None, f"failed to parse {table_title}"
    assert match.group("year") == str(year), f"{match.group('year')} != {year}"

    # CLAUDE: disable this warning
    # "WARNING:fastexcel.types.dtype:Could not determine dtype for column 7, falling back to string"
    table_contents = xl.load_sheet(0, header_row=2 if year > 2010 else 6).to_polars()

    return (
        table_contents.rename(
            {"__UNNAMED__0": "age", "Age (years)": "age", "Age": "age"},
            strict=False,
        )
        .rename(lambda s: s.strip().replace("x", "").replace("()", ""))
        .select(
            pl.lit(table_title).alias("table_title"),
            *(pl.lit(v.strip()).alias(k) for k, v in match.groupdict().items()),
            pl.col("age").str.extract(r"^(\d+).*$").str.to_integer(),
            *LIFE_TABLE_COLS,
        )
        .drop_nulls("age")
    )


if __name__ == "__main__":
    df = get_all_life_tables()
