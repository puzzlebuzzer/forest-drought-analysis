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


def add_date_range_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--start-date",
        default="2017-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        default="2026-03-01",
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