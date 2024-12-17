import os
import io
import time
import pstats
import cProfile
from pstats import SortKey
from example_inference import main


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


def profiled_main():
    return profile_function(main)


if __name__ == "__main__":
    profiled_main()
