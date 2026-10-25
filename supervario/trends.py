"""Legacy String Theory statistics parser; no longer used by automatic fitting."""

from collections import Counter

import numpy as np

from .data import chunks
from .local_tools import filters


DEPRECATION_MESSAGE = (
    "String Theory orientation import is deprecated and ignored. "
    "SuperVario now screens supported 3D orthogonal frames directly from sample pairs. "
    "Path frequencies and interpolated point clouds are not variogram observations."
)


def string_theory_candidates(path: str, domain: str):
    counts = Counter()
    accepted = 0
    helper = filters()
    for frame in chunks({"path": path}, ["Azimuth", "Dip", "Category", "Accepted"], text_columns=["Category", "Accepted"]):
        selected = frame["Category"].map(helper.norm_token) == domain
        for az, dip, keep in frame.loc[selected, ["Azimuth", "Dip", "Accepted"]].itertuples(index=False, name=None):
            token = str(keep).strip().lower()
            if token not in ("true", "false", "1", "0"):
                raise ValueError(f"Unrecognised String Theory Accepted value: {keep}")
            if token in ("false", "0"):
                continue
            az, dip = float(az), float(dip)
            if not np.isfinite([az, dip]).all() or not 0 <= dip <= 90 or not 0 <= az <= 360:
                raise ValueError("String Theory orientations require azimuth 0..360 and downward dip 0..90.")
            # Axial horizontal directions are equivalent modulo 180; inclined
            # lines retain their full downward azimuth.
            if dip < 5:
                az %= 180
            counts[(int((az % 360) // 10), min(int(dip // 10), 8))] += 1
            accepted += 1
    if not accepted:
        raise ValueError(
            f"No accepted String Theory paths match Category '{domain}'. "
            "Use per-domain path_stats exports or clear the optional trend file."
        )
    ranked = counts.most_common(12)
    candidates = [{"azimuth": az * 10 + 5, "plunge": dip * 10 + 5, "count": count}
                  for (az, dip), count in ranked]
    return candidates, {
        "source": path, "category": domain, "accepted_paths": accepted, "candidates": candidates,
        "method": "10-degree joint orientation histogram; top 12 bins supplement the continuity search",
        "caution": "Connection frequencies depend on sampling and String Theory settings; path lengths are not variogram ranges.",
    }
