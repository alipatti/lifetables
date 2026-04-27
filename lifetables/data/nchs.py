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

VOLUMES = {
    2019: "70-19",
    2020: "71-01",
    2021: "72-12",
    2022: "74-02",
    2023: "74-06",
}


BASE_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR"
CLIENT = SyncCacheClient(base_url=BASE_URL)


RACE_MAPPING = {
    # total
    "": "Pooled",
    "the total population": "Pooled",
    # hispanic
    "hispanic": "Hispanic",
    "the hispanic population": "Hispanic",
    # white
    "white, non-hispanic": "White",
    "non-hispanic white": "White",
    "the white, non-hispanic population": "White",
    "the non-hispanic white population": "White",
    # black
    "black, non-hispanic": "Black",
    "non-hispanic black": "Black",
    "the black, non-hispanic population": "Black",
    "the non-hispanic black population": "Black",
    # asian
    "asian, non-hispanic": "Asian",
    "non-hispanic asian": "Asian",
    "the asian, non-hispanic population": "Asian",
    "the non-hispanic asian population": "Asian",
    # aian
    "american indian and alaska native, non-hispanic": "AIAN",
    "non-hispanic american indian or alaska native": "AIAN",
    "the american indian and alaska native, non-hispanic population": "AIAN",
    "the non-hispanic american indian or alaska native population": "AIAN",
}


def get_all_life_tables() -> pl.DataFrame:
    return pl.concat(
        get_life_table(year, table) for year in VOLUMES.keys() for table in range(1, 19)
    ).with_columns(
        pl.col("year").str.to_integer(),
        pl.col("sex").replace_strict(
            {"females": "Female", "males": "Male", "": "Pooled"}
        ),
        pl.col("race").str.to_lowercase().replace_strict(RACE_MAPPING),
    )


def get_life_table(year: int, table_number: int) -> pl.DataFrame:

    logging.info(f"Fetching Table {table_number} for {year}")

    release = VOLUMES.get(year)
    assert release, f"no volume for {year}"

    url = f"{BASE_URL}/{release}/Table{table_number:>02}.xlsx"
    response = CLIENT.get(url)
    response.raise_for_status()

    xl = fastexcel.read_excel(response.content)

    # get the table title from cell AA
    table_title = xl.load_sheet(0, n_rows=1).to_polars().columns[0].strip()

    match = re.match(LIFE_TABLE_TITLE_PATTERN, table_title)
    assert match is not None, f"failed to parse {table_title}"
    assert match.group("year") == str(year), f"{match.group('year')} != {year}"

    return (
        xl.load_sheet(0, header_row=2)
        .to_polars()
        .rename({"__UNNAMED__0": "age"})
        .rename(lambda s: s.replace("x", ""))
        .select(
            pl.lit(table_title).alias("table_title"),
            *(pl.lit(v.strip()).alias(k) for k, v in match.groupdict().items()),
            pl.col("age").str.extract(r"^(\d+).*$").str.to_integer(),
            pl.col("^.$"),
        )
        .drop_nulls("age")
    )


if __name__ == "__main__":
    df = get_all_life_tables()
