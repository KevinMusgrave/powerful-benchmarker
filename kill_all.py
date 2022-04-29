import argparse

from powerful_benchmarker.utils.constants import JOBIDS_FILENAME, add_default_args
from powerful_benchmarker.utils.utils import kill_all_jobs
from validator_tests.utils.constants import JOBIDS_FILENAME as V_JOBSID_FILENAME

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_default_args(parser, ["exp_folder"])
    parser.add_argument("--validator_tests", action="store_true")
    parser.add_argument("--old_format", action="store_true")
    args = parser.parse_args()
    filename = V_JOBSID_FILENAME if args.validator_tests else JOBIDS_FILENAME
    if args.old_format:
        filename = filename.replace(".json", ".txt")
    kill_all_jobs(args.exp_folder, filename)
