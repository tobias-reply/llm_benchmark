#!/usr/bin/env python3
"""
Script to run benchmark 10 times with 30-minute intervals
Cycles through temperature values from 0.1 to 1.0
Usage: AWS_PROFILE=tobi-default uv run python run_benchmark_schedule.py
"""

import subprocess
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path


# Configuration
TOTAL_RUNS = 9
INTERVAL_MINUTES = 30
NUMBER_OF_CALLS = 100
TEMPERATURES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BENCHMARK_SCRIPT = "src/benchmark/main.py"


def print_colored(message: str, color: str = "blue"):
    """Print colored output"""
    colors = {
        "blue": "\033[0;34m",
        "green": "\033[0;32m",
        "yellow": "\033[1;33m",
        "red": "\033[0;31m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['blue'])}{message}{colors['reset']}")


def print_header():
    """Print script header"""
    print_colored("=" * 70, "blue")
    print_colored("Benchmark Scheduler - Temperature Sweep", "blue")
    print_colored("=" * 70, "blue")
    print(f"Total runs: {TOTAL_RUNS}")
    print(f"Interval: {INTERVAL_MINUTES} minutes")
    print(f"Calls per run: {NUMBER_OF_CALLS}")
    print(f"Temperatures: {', '.join(str(t) for t in TEMPERATURES)}")
    print_colored("=" * 70 + "\n", "blue")


def run_benchmark(run_number: int, temperature: float) -> bool:
    """Run a single benchmark iteration with specified temperature"""
    start_time = datetime.now()
    print_colored(
        f"[Run {run_number}/{TOTAL_RUNS}] Starting at {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"with temperature={temperature}",
        "blue"
    )

    try:
        # Run from benchmark root directory (where config/ is located)
        benchmark_dir = Path(__file__).parent

        # Run the benchmark using uv with temperature parameter
        result = subprocess.run(
            [
                "uv", "run", "python",
                BENCHMARK_SCRIPT,
                "--number-of-calls", str(NUMBER_OF_CALLS),
                "--temperature", str(temperature)
            ],
            cwd=benchmark_dir,
            check=True,
            capture_output=False  # Show output in real-time
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        print_colored(
            f"[Run {run_number}/{TOTAL_RUNS}] Completed successfully at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(Temperature: {temperature}, Duration: {duration:.1f} minutes)\n",
            "green"
        )
        return True

    except subprocess.CalledProcessError as e:
        end_time = datetime.now()
        print_colored(
            f"[Run {run_number}/{TOTAL_RUNS}] Failed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(Temperature: {temperature})",
            "red"
        )
        print_colored(f"Error: {e}\n", "red")
        return False
    except Exception as e:
        end_time = datetime.now()
        print_colored(
            f"[Run {run_number}/{TOTAL_RUNS}] Error at {end_time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(Temperature: {temperature})",
            "red"
        )
        print_colored(f"Error: {e}\n", "red")
        return False


def display_next_run(run_number: int, next_temperature: float):
    """Display information about the next scheduled run"""
    next_run_time = datetime.now() + timedelta(minutes=INTERVAL_MINUTES)
    print_colored(
        f"Next run ({run_number}/{TOTAL_RUNS}) scheduled at: {next_run_time.strftime('%Y-%m-%d %H:%M:%S')} "
        f"with temperature={next_temperature}",
        "yellow"
    )
    print_colored(f"Waiting {INTERVAL_MINUTES} minutes...\n", "yellow")


def main():
    """Main execution function"""
    print_header()

    successful_runs = 0
    failed_runs = 0
    results_summary = []

    for i in range(1, TOTAL_RUNS + 1):
        # Get temperature for this run (0-indexed)
        temperature = TEMPERATURES[i - 1]

        success = run_benchmark(i, temperature)

        results_summary.append({
            "run": i,
            "temperature": temperature,
            "success": success
        })

        if success:
            successful_runs += 1
        else:
            failed_runs += 1

        # Sleep between runs (except after the last run)
        if i < TOTAL_RUNS:
            next_temperature = TEMPERATURES[i]
            display_next_run(i + 1, next_temperature)

            # Sleep with progress indicator
            for remaining in range(INTERVAL_MINUTES * 60, 0, -60):
                time.sleep(60)
                mins_remaining = remaining // 60
                if mins_remaining > 0:
                    print(f"  Time until next run: {mins_remaining} minutes remaining...", end="\r")
            print()  # New line after progress

    # Print final summary
    print_colored("=" * 70, "green")
    print_colored(f"All {TOTAL_RUNS} benchmark runs completed!", "green")
    print_colored(f"Successful: {successful_runs} | Failed: {failed_runs}", "green")
    print_colored("=" * 70, "green")
    print("\nRun Summary:")
    for result in results_summary:
        status = "✓" if result["success"] else "✗"
        color = "green" if result["success"] else "red"
        print_colored(
            f"  Run {result['run']}: Temperature {result['temperature']:.1f} - {status}",
            color
        )
    print_colored("=" * 70, "green")

    # Exit with error code if any runs failed
    sys.exit(0 if failed_runs == 0 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\n\nScript interrupted by user", "yellow")
        sys.exit(130)