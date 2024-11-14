from argparse import ArgumentParser
from pathlib import Path
from typing import List

import ray
from tqdm import tqdm

from process.av2_processor import Av2Extractor
from utils.ray_utils import ActorHandle, ProgressBar


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


@ray.remote
def preprocess_batch(extractor: Av2Extractor, file_list: List[Path], pb: ActorHandle):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val"]:       # No test set
        save_dir = data_root / "processed" / mode
        extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
            pb_actor = pb.actor

            for i in range(0, len(scenario_files), batch):
                preprocess_batch.remote(extractor, scenario_files[i : i + batch], pb_actor)
            pb.print_until_done()
            
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--multiagent", "-m", action="store_true")
    parser.add_argument("--threads", "-t", type=int, default=16)

    args = parser.parse_args()
    ray.init(num_cpus=args.threads)
    preprocess(args)