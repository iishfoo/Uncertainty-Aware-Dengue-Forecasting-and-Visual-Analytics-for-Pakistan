"""
Run the whole pipeline in the right order and stop at the first error.

    python run_all.py              # everything (the GRU step takes several minutes)
    python run_all.py --skip-gru   # quick check without the GRU

Order: parse_bulletins -> build_dataset -> coverage_report -> run_models -> run_gru -> summarize_results
Everything printed is also saved to results/run_log.txt  (paste that file to me when you want a check).
"""
import subprocess
import sys
from pathlib import Path

STEPS = ["parse_bulletins.py", "build_dataset.py", "coverage_report.py", "run_models.py", "run_gru.py",
         "summarize_results.py"]


def main():
    steps = [s for s in STEPS if not (s == "run_gru.py" and "--skip-gru" in sys.argv)]
    Path("results").mkdir(exist_ok=True)
    with open("results/run_log.txt", "w", encoding="utf-8") as log:
        for s in steps:
            head = f"\n{'=' * 70}\n>>> {s}\n{'=' * 70}\n"
            print(head, end="")
            log.write(head)
            p = subprocess.Popen([sys.executable, s], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
            for line in p.stdout:
                if "\r" in line:                       # progress bars: keep only the last update of the line
                    parts = [x for x in line.rstrip("\r\n").split("\r") if x.strip()]
                    line = (parts[-1] if parts else "") + "\n"
                print(line, end="")
                log.write(line)
            if p.wait() != 0:
                msg = f"\nSTOPPED: {s} failed (exit code {p.returncode}). Fix this first, then run again.\n"
                print(msg)
                log.write(msg)
                sys.exit(1)
    print("\nAll steps finished. Full output: results/run_log.txt")


if __name__ == "__main__":
    main()
