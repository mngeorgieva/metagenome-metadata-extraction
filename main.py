# /// script
# dependencies = [
#     "click>=8.3.0",
#     "polars>=1.33.1",
# ]
# ///

"""Harmonise metagenome summary tables from different formats into a standard format."""

import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from pathlib import Path

import click
import polars as pl
from polars import DataFrame, Expr, Series
from polars.datatypes import Float32, Int32, String

# Constants
LOOKUP_TABLE_SCHEMA = {
    "phylum": String,
    "reducing_env": String,
    "is_deep_sea": String,
    "host_taxname": String,
    "host_tolid": String,
}


# Data models
class SummaryTable(ABC):
    """An abstract base class for metagenome summary tables."""

    HARMONISED_SCHEMA = {
        "host_tolid": String,
        "host_species": String,
        "assembler": String,
        "binner/refiner": String,
        "size": Int32,
        "quality": String,
        "completeness": Float32,
        "contamination": Float32,
        "gtdb_classification": String,
        "ncbi_classification": String,
        "phylum": String,
        "reducing_env": String,
        "is_deep_sea": String,
    }

    def __init__(self, input: DataFrame, lookup_table: DataFrame):
        self.df = input
        self.lut = lookup_table

    @abstractmethod
    def host_tolid(self) -> Series:
        pass

    @abstractmethod
    def host_species(self) -> Series:
        pass

    @abstractmethod
    def assembler(self) -> Series:
        pass

    @abstractmethod
    def binner_refiner(self) -> Series:
        pass

    @abstractmethod
    def size(self) -> Series:
        pass

    @abstractmethod
    def quality(self) -> Series:
        pass

    @abstractmethod
    def completeness(self) -> Series:
        pass

    @abstractmethod
    def contamination(self) -> Series:
        pass

    @abstractmethod
    def gtdb_classification(self) -> Series:
        pass

    @abstractmethod
    def ncbi_classification(self) -> Series:
        pass

    def filter(self) -> Expr | None:
        """Apply a filter to the harmonised DataFrame."""
        return None

    def drop(self) -> list[str] | None:
        """Drop columns from the harmonised DataFrame."""
        return None

    def rename_columns(self) -> dict | None:
        """Rename columns in the harmonised DataFrame."""
        return None

    def harmonise(self) -> DataFrame:
        """Harmonise the summary table to the standard schema."""
        data = {
            "host_tolid": self.host_tolid(),
            "host_species": self.host_species(),
            "assembler": self.assembler(),
            "binner/refiner": self.binner_refiner(),
            "size": self.size(),
            "quality": self.quality(),
            "completeness": self.completeness(),
            "contamination": self.contamination(),
            "gtdb_classification": self.gtdb_classification(),
            "ncbi_classification": self.ncbi_classification(),
        }
        try:
            df = pl.DataFrame(data).join(
                self.lut, on="host_tolid", how="left", validate="m:1"
            )
        except pl.exceptions.ComputeError as e:
            raise ValueError(
                "Error joining summary table with lookup table. "
                "Please ensure the lookup table contains unique host_tolid values."
            ) from e
        df = df.cast(self.HARMONISED_SCHEMA)
        df = df.drop(self.drop()) if self.drop() is not None else df
        df = (
            df.rename(self.rename_columns())
            if self.rename_columns() is not None
            else df
        )
        df = df.filter(self.filter()) if self.filter() is not None else df
        return df.select(self.HARMONISED_SCHEMA.keys())


class Jim(SummaryTable):
    """Jim's metagenome summary table format."""

    def __init__(self, input: DataFrame, lookup_table: DataFrame):
        super().__init__(input, lookup_table)

    def host_tolid(self) -> Series:
        return self.df["bin"].str.split("_").list.get(0)

    def host_species(self) -> Series:
        return Series([None] * self.df.height)

    def assembler(self) -> Series:
        return self.df["assembler"]

    def binner_refiner(self) -> Series:
        return self.df["binner"]

    def size(self) -> Series:
        return self.df["sum_len"]

    def quality(self) -> Series:
        return self.df["quality"]

    def completeness(self) -> Series:
        return self.df["completeness"]

    def contamination(self) -> Series:
        return self.df["contamination"]

    def gtdb_classification(self) -> Series:
        return self.df["gtdb_classification"]

    def ncbi_classification(self) -> Series:
        return self.df["ncbi_classification"]

    def filter(self) -> Expr:
        return (pl.col("binner/refiner") == "dastool") & (
            pl.col("quality").is_in(["high", "medium"])
        )

    def drop(self) -> list[str]:
        return ["host_species"]

    def rename_columns(self) -> dict:
        return {"host_taxname": "host_species"}


class Noah(SummaryTable):
    """Noah's metagenome summary table format."""

    def __init__(self, input: DataFrame, lookup_table: DataFrame):
        super().__init__(input, lookup_table)

    def host_tolid(self) -> Series:
        return self.df["host"]

    def host_species(self) -> Series:
        return self.df["host_species"]

    def assembler(self) -> Series:
        return self.df["assembler"]

    def binner_refiner(self) -> Series:
        return self.df["refining_program"]

    def size(self) -> Series:
        return self.df["size"]

    def quality(self) -> Series:
        return self.df["quality"].str.to_lowercase()

    def completeness(self) -> Series:
        return self.df["Completeness"]

    def contamination(self) -> Series:
        return self.df["Contamination"]

    def gtdb_classification(self) -> Series:
        return self.df["classification"]

    def ncbi_classification(self) -> Series:
        return self.df["ncbi_classification"]

    def filter(self) -> Expr:
        return pl.col("quality").is_in(["high", "medium"])


class SummaryTableFormat(Enum):
    """An enumeration of metagenome summary table formats."""

    JIM = ("\t", Jim)
    NOAH = (",", Noah)

    def __init__(self, separator: str, table_definition: SummaryTable):
        self._separator = separator
        self._table_definition = table_definition

    @property
    def separator(self) -> str:
        return self._separator

    def table_definition(
        self, table_data=DataFrame, lookup_table=DataFrame
    ) -> SummaryTable:
        return self._table_definition(table_data, lookup_table)


OuputOptions = namedtuple("OuputOptions", ["write_mode", "add_header"])


# Functions


def get_lookup_table(file_path: Path) -> DataFrame:
    """Get a lookup table as a Polars DataFrame.

    Args:
        file_path (Path): Path to the lookup table CSV file.

    Returns:
        pl.DataFrame: The lookup table as a Polars DataFrame.
    """
    return pl.read_csv(
        file_path,
        columns=[c for c in LOOKUP_TABLE_SCHEMA.keys()],
        schema_overrides=LOOKUP_TABLE_SCHEMA,
    )


def expand_path(path_pattern: str) -> Path | None:
    """Expand a file path pattern with wildcards and return the first matching path.

    Args:
        path_pattern (str): The file path pattern with wildcards.

    Returns:
        Path | None: The first matching file path, or None if no match is found.
    """
    p = Path(path_pattern).expanduser()
    parts = p.parts[p.is_absolute() :]
    paths = Path(p.root).glob(str(Path(*parts)))
    return next(paths, None)


def get_summary_table_locations(file_path: Path) -> list[Path]:
    """Get a list of metagenome summary table file paths from a text file.

    Args:
        file_path (Path): Path to the text file containing the list of metagenome summary table paths.

    Returns:
        List[Path]: A list of metagenome summary table file paths.
    """
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    locations = []
    not_found = []
    for line in lines:
        expanded_path = expand_path(line)
        if expanded_path is not None:
            locations.append(expanded_path)
        else:
            not_found.append(line)
    if not_found:
        sys.exit(
            f"Did nothing. The following paths were not found:\n{'\n'.join(not_found)}"
        )
    return locations


def get_summary_table(
    file_path: Path, format=SummaryTableFormat, lookup_table=DataFrame
) -> DataFrame:
    """Get a metagenome summary table as a Polars DataFrame.

    Args:
        file_path (Path): Path to the metagenome summary table file.
        format (SummaryTableFormat): The format of the metagenome summary table.

    Returns:
        pl.DataFrame: The metagenome summary table as a Polars DataFrame.

    Raises:
        ValueError: If there is an error reading the summary table or if required columns are missing
    """
    try:
        df = pl.read_csv(file_path, separator=format.separator)
        df = format.table_definition(df, lookup_table).harmonise()
    except pl.exceptions.ColumnNotFoundError as e:
        raise ValueError(
            f"Error reading summary table {file_path}. "
            "Please ensure the file has the correct format and all required columns are present."
        ) from e
    return df


# CLI
@click.command()
@click.option(
    "--input",
    type=click.Path(exists=True),
    prompt="Input text file contain the list of newline-separated metagenome summary table paths",
)
@click.option(
    "--output",
    type=click.Path(writable=True, file_okay=True, dir_okay=True),
    prompt="CSV file to append the results to",
)
@click.option(
    "--lut",
    type=click.Path(exists=True),
    prompt="Lookup table CSV file containing the tol id to sample name mapping",
)
@click.option(
    "--format",
    default="jim",
    show_default=True,
    help="The format of the metagenome summary table.",
    type=click.Choice(SummaryTableFormat, case_sensitive=False),
)
def main(input, output, format, lut):
    """Provide a list of newline separated metagenome summary table paths (INPUT),
    a format type (`jim` or `noah`), harmonize the summary tables
    and append them to a CSV file (OUTPUT).
    """
    print(f"Input file: {input}")
    print(f"Output file: {output}")
    print(f"Lookup table file: {lut}")
    print(f"Format: {format.name}")

    summary_table_paths = get_summary_table_locations(Path(input))
    print(f"Found {len(summary_table_paths)} summary table paths.")

    output_options = (
        OuputOptions("w", True)
        if not Path(output).exists()
        else OuputOptions("a", False)
    )
    print(f"Write mode: {output_options.write_mode}")

    for path in summary_table_paths:
        df = get_summary_table(
            path, format=format, lookup_table=get_lookup_table(Path(lut))
        )
        with open(output, output_options.write_mode) as f:
            df.write_csv(f, include_header=output_options.add_header)
        print(f"Processed {path}. {df.height} rows written to {output}")


if __name__ == "__main__":
    main()
