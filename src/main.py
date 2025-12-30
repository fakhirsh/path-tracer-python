
from core.material import *
from scenes import *
from util import *
from core import *
import cProfile
import pstats

#------------------------------------------------------------------------

def main():
    vol2_sec2_6_interactive()

#------------------------------------------------------------------------

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # Save profile data to file for visualization tools
    profiler.dump_stats('../temp/profile_output.prof')

    # # Print profiling results to console
    # stats = pstats.Stats(profiler)
    # stats.sort_stats('cumulative')

    # print("\n" + "="*80)
    # print("PROFILING RESULTS - Top 50 functions by cumulative time")
    # print("="*80)
    # stats.print_stats(50)

    # print("\n" + "="*80)
    # print("PROFILING RESULTS - Top 50 functions by total time")
    # print("="*80)
    # stats.sort_stats('tottime')
    # stats.print_stats(50)