from pathlib import Path
from typing import Tuple, Literal, Final, List

from av2.datasets.motion_forecasting.data_schema import ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneType
import pandas as pd


KEYS: Final[List[str]] = [
    "actor_pos",
    "actor_attr",
    "actor_real_pos",
    "actor_centers",
    "actor_angles",
    "actor_velocity",
    "actor_velocity_diff",
    "lane_positions",
    "lane_centers",
    "lane_angles",
    "lane_attr",
    "is_intersections",
]


def load_scenario(file_path: Path) -> Tuple[pd.DataFrame, ArgoverseStaticMap, str]:
    scenario_id = file_path.stem.split("_")[-1]
    scenario_df = pd.read_parquet(file_path)

    scenario_map = ArgoverseStaticMap.from_json(
        file_path.parents[0] / 
        f"log_map_archive_{scenario_id}.json"
    )

    return scenario_df, scenario_map, scenario_id


def object_type_to_int(object_name: str, type_name: Literal['o2i', 'o2if', 'l2i', 'unk']) -> int:
    object_to_int = {
        ObjectType.VEHICLE: 0,
        ObjectType.PEDESTRIAN: 1,
        ObjectType.MOTORCYCLIST: 2,
        ObjectType.CYCLIST: 3,
        ObjectType.BUS: 4,
        ObjectType.STATIC: 5,
        ObjectType.BACKGROUND: 6,
        ObjectType.CONSTRUCTION: 7,
        ObjectType.RIDERLESS_BICYCLE: 8,
        ObjectType.UNKNOWN: 9,
    }

    object_to_int_fixed = {
        ObjectType.VEHICLE: 0,
        ObjectType.PEDESTRIAN: 1,
        ObjectType.MOTORCYCLIST: 2,
        ObjectType.CYCLIST: 2,
        ObjectType.BUS: 0,
        ObjectType.STATIC: 3,
        ObjectType.BACKGROUND: 3,
        ObjectType.CONSTRUCTION: 3,
        ObjectType.RIDERLESS_BICYCLE: 3,
        ObjectType.UNKNOWN: 3,      
    }

    lane_to_int = {
        LaneType.VEHICLE: 0,
        LaneType.BIKE: 1,
        LaneType.BUS: 2,
    }

    if type_name == 'o2i':
        return object_to_int[object_name]
    elif type_name == 'o2if':
        return object_to_int_fixed[object_name]
    elif type_name == 'l2i':
        return lane_to_int[object_name]
    else:
        raise NotImplementedError('no such transfer rule')

