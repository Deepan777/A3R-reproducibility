"""Parser test on a SYNTHETIC fixture (format check only; not measurement data)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_runtime_logs import parse_file, summarise  # noqa: E402

FIXTURE_LINE = ("01-01 00:00:00.000  1234  1250 I Unity   : [A3R_BENCHMARK] t={t:.1f}s | RenderFps=59.0 | "
                "CamFps=30.0 | MpFps=4.0 | MpLat=25.0ms | ScorerMeanLat=0.50ms | ScorerP95Lat=1.00ms | "
                "PredLat=0.030ms | DropInf=0 | DropScore=0 | Face=True | HeapMB=1.70")


def test_parse_and_summarise(tmp_path):
    for rep in (1, 2, 3):
        p = tmp_path / f"run_FIXTURE_combined_{rep}.txt"
        p.write_text("\n".join(FIXTURE_LINE.format(t=t) for t in range(5, 125, 5)) + "\nunrelated line\n")
    import pandas as pd
    win = pd.concat([parse_file(p) for p in sorted(tmp_path.glob("run_*.txt"))])
    assert len(win) == 3 * 24 and set(win.device) == {"FIXTURE"}
    s = summarise(win)
    assert s.loc[0, "n_runs"] == 3 and s.loc[0, "mp_lat_ms_median"] == 25.0
