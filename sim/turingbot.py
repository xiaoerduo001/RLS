"""Call the local TuringBot command line to write a formula for one (x, y) segment."""

import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fitters import SegmentFit


@dataclass(frozen=True)
class TuringBotConfig:
    binary: str = "/Applications/TuringBot.app/Contents/MacOS/TuringBot"
    seconds: float = 8.0
    threads: int = 2
    search_metric: int = 4  # RMS
    maximum_formula_complexity: int = 20
    history_size: int = 0
    allowed_functions: str = "+ *"


class TuringBotFitter:
    """Search for a fixed number of seconds and keep the formula with the smallest RMS. The process group is stopped at the deadline."""

    def __init__(self, search):
        self.search = search
        self.work = Path(tempfile.mkdtemp(prefix="turingbot_"))
        binary = Path(search.binary)
        if not binary.is_file():
            raise FileNotFoundError(f"TuringBot not found: {binary}")

    def close(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def license_status(self):
        proc = subprocess.run(
            [self.search.binary, "--license-status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (proc.stdout or proc.stderr or "").strip()

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        data_path = self.work / "input.csv"
        out_path = self.work / "solutions.txt"
        if out_path.exists():
            out_path.unlink()
        _write_xy(data_path, x, y)

        cmd = [
            self.search.binary,
            str(data_path),
            "--outfile",
            str(out_path),
            "--threads",
            str(self.search.threads),
            "--search-metric",
            str(self.search.search_metric),
            "--maximum-formula-complexity",
            str(self.search.maximum_formula_complexity),
            "--history-size",
            str(self.search.history_size),
            "--allowed-functions",
            self.search.allowed_functions,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _stderr = proc.communicate(timeout=self.search.seconds)
        except subprocess.TimeoutExpired:
            _stop_process_group(proc)
            stdout, _stderr = proc.communicate(timeout=10)
        text = ""
        if out_path.exists():
            text = out_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            text = stdout or ""
        fits = parse_solution_text(text)
        if not fits:
            return SegmentFit(formula="", error=np.inf, size=None)
        return min(fits, key=lambda item: (item.error, item.size or 0))


def parse_solution_text(text):
    """Parse the last 'Formulas generated' block. Each row is complexity, RMS, formula."""
    rows = []
    pattern = re.compile(
        r"^(\d+)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+(\S.*?)\s*$"
    )
    blocks = re.split(r"Formulas generated:\s*\d+\s*", text)
    body = blocks[-1] if blocks else text
    for line in body.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        rows.append(
            SegmentFit(
                formula=match.group(3).strip(),
                error=float(match.group(2)),
                size=int(match.group(1)),
            )
        )
    return rows


def _write_xy(path, x, y):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("x,y\n")
        for xv, yv in zip(x, y):
            handle.write(f"{float(xv):.10g},{float(yv):.10g}\n")


def _stop_process_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 3
    while time.time() < deadline and proc.poll() is None:
        time.sleep(0.1)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
