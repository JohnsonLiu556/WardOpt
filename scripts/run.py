# run.py — tiny pipeline runner
# Usage:
#   python run.py --step forecast
#   python run.py --step optimize
#   python run.py --step insights
#   python run.py --step all   (default)

import argparse, subprocess, sys

def sh(cmd: str):
    print(">>", cmd)
    subprocess.check_call(cmd, shell=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["forecast", "optimize", "insights", "all"], default="all")
    args = ap.parse_args()

    # run whichever pieces you want
    if args.step in ("forecast", "all"):
        sh(f"{sys.executable} forecast.py")
    if args.step in ("optimize", "all"):
        sh(f"{sys.executable} optimize.py")
    if args.step in ("insights", "all"):
        sh(f"{sys.executable} insights.py")

if __name__ == "__main__":
    main()
