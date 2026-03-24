import argparse


def make_parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=description)


def add_aoi_arg(
    parser: argparse.ArgumentParser,
    required: bool = False,
    default: str | None = "north",
) -> None:
    parser.add_argument(
        "-a",
        "--aoi",
        choices=["north", "south"],
        required=required,
        default=None if required else default,
        help="AOI to use",
    )


def add_index_arg(
    parser: argparse.ArgumentParser,
    required: bool = False,
    default: str | None = "NDVI",
) -> None:
    parser.add_argument(
        "-i",
        "--index",
        choices=["NDVI", "NDMI", "EVI"],
        required=required,
        default=None if required else default,
        help="Index to analyze",
    )


def add_indices_arg(
    parser: argparse.ArgumentParser,
    required: bool = False,
    default: list[str] | None = None,
) -> None:
    parser.add_argument(
        "--indices",
        nargs="+",
        choices=["NDVI", "NDMI", "EVI"],
        required=required,
        default=default if default is not None else ["NDVI", "NDMI", "EVI"],
        help="Indices to process",
    )


def add_date_range_args(
    parser: argparse.ArgumentParser,
    default_start: str = "2017-01-01",
    default_end: str = "2026-03-01",
) -> None:
    parser.add_argument(
        "--start-date",
        default=default_start,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        default=default_end,
        help="End date in YYYY-MM-DD format",
    )


def add_cloud_arg(
    parser: argparse.ArgumentParser,
    default: int = 40,
) -> None:
    parser.add_argument(
        "--cloud-max",
        type=int,
        default=default,
        help="Maximum cloud cover percentage",
    )


def add_cache_suffix_arg(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--cache-suffix",
        default=None,
        metavar="SUFFIX",
        help=(
            "Append SUFFIX to the cache root folder name, creating an adjacent "
            "dated cache alongside the original.  E.g. --cache-suffix _3_24 "
            "writes to GWNF_cache_3_24/ instead of GWNF_cache/."
        ),
    )