import os
import io
import time
import pstats
import argparse
import cProfile
from pstats import SortKey


def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    directory_path = "profiling_results"
    os.makedirs(directory_path, exist_ok=True)
    profiler.dump_stats(f'{directory_path}/{time.strftime("%Y%m%d-%H%M%S")}.prof')
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(50)
    print(s.getvalue())
    return result


def profiled_main(model_version):
    if model_version == "RMBG-1.4":
        from inference_1_4 import main

        return profile_function(main)
    elif model_version == "RMBG-2.0":
        from inference_2_0 import main

        return profile_function(main)
    else:
        raise ValueError(f"Invalid model version: {model_version}")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-version",
        "-m",
        type=str,
        default="RMBG-1.4",
        choices=["RMBG-1.4", "RMBG-2.0"],
        help="Matting method to use (default: 'RMBG-1.4')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    profiled_main(args.model_version)
