from pathlib import Path
import argparse

from .boosting import run_boosting
from .utils import load_config

def main():
    parser = argparse.ArgumentParser()
    default_conf = Path(__file__).resolve().parent / 'conf' / 'conf.yaml'
    parser.add_argument(
        '-c', '--conf_dir',
        default=str(default_conf),
        help='Path to the configuration file',
    )
    args = parser.parse_args()

    config = load_config(args.conf_dir)
    run_boosting(config)

if __name__ == '__main__':
    main()