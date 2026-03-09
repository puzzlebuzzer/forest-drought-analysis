from dataclasses import dataclass
from pathlib import Path

from src.paths import project_path


@dataclass(frozen=True)
class AOIConfig:
    key: str
    landscape_id: str
    cache_root: Path
    index_cache_root: Path
    terrain_dir: Path
    forest_group_dir: Path
    forest_type_dir: Path
    ecozone_dir: Path
    species_raster: Path
    raw_ecozone_raster: Path


_AOI_CONFIGS = {
    "north": AOIConfig(
        key="north",
        landscape_id="UW-GW North",
        cache_root=project_path("north_cache_root"),
        index_cache_root=project_path("north_index_cache_root"),
        terrain_dir=project_path("north_terrain_dir"),
        forest_group_dir=project_path("north_forest_group_dir"),
        forest_type_dir=project_path("north_forest_type_dir"),
        ecozone_dir=project_path("north_ecozone_dir"),
        species_raster=project_path("north_species_raster"),
        raw_ecozone_raster=project_path("north_raw_ecozone_raster"),
    ),
    "south": AOIConfig(
        key="south",
        landscape_id="UW-Smoky",
        cache_root=project_path("south_cache_root"),
        index_cache_root=project_path("south_index_cache_root"),
        terrain_dir=project_path("south_terrain_dir"),
        forest_group_dir=project_path("south_forest_group_dir"),
        forest_type_dir=project_path("south_forest_type_dir"),
        ecozone_dir=project_path("south_ecozone_dir"),
        species_raster=project_path("south_species_raster"),
        raw_ecozone_raster=project_path("south_raw_ecozone_raster"),
    ),
}


def valid_aois() -> list[str]:
    return list(_AOI_CONFIGS.keys())


def get_aoi_config(aoi: str) -> AOIConfig:
    key = aoi.lower()
    try:
        return _AOI_CONFIGS[key]
    except KeyError as e:
        raise ValueError(
            f"Unknown AOI '{aoi}'. Valid options: {valid_aois()}"
        ) from e


def get_aoi_shapefile() -> Path:
    return project_path("tnc_aoi_shapefile")


def get_forest_group_inventory_path(aoi: str) -> Path:
    cfg = get_aoi_config(aoi)
    return cfg.forest_group_dir / "forest_group_inventory.json"