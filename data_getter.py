"""Download the Ukrainian air-raid alert log and count alerts per day and per week.

Each alert is counted once, however many rows the log has for it.

What counts as one alert
------------------------
* Regional alert: an unbroken period during which an alert was in force in an
  oblast (region), either for the whole oblast or for any of its raions
  (districts). Rows of the same oblast whose time spans overlap are one alert,
  so an alert logged for the oblast and again for each of its raions counts
  once. Since late 2025 alerts are declared per raion; raion alerts of one
  oblast that overlap in time are likewise one alert, which keeps the counts
  comparable with the earlier oblast-wide period.
* Local alert: an alert declared for a single hromada (municipality). It is
  counted only if it starts while no regional alert is in force in its oblast;
  otherwise it is part of that regional alert. Local alerts are merged per
  hromada and kept out of the regional chain, because a few front-line hromadas
  are under alert for months at a time and would otherwise swallow every other
  alert of their oblast.

The national figure for a day is the number of alerts that started that day
(Kyiv time), summed over the 24 oblasts and Kyiv city. ``alerts`` is the total;
``regional_alerts`` and ``local_alerts`` are its two parts. Published statistics
count regional alerts only.

Time under alert
----------------
A count says nothing about how long alerts last, and over the war alerts have
grown longer much more than they have grown more frequent. ``alert_hours``
measures that: the hours a district (raion) spent under alert, averaged over all
districts. A district is under alert while an alert is in force for the district
itself or for its whole oblast. Kyiv city has no raions and counts as one
district. Alerts for single hromadas are left out, because they cover only part
of a district. Luhansk oblast and Crimea are left out, because their permanent
alerts are not in the log.

Pipeline
--------
1. Download the official alert log from Vadimkin/ukrainian-air-raid-sirens-dataset
   into ``data/raw/``.
2. Clean it: drop exact duplicate rows, parse timestamps (UTC).
3. Build regional alerts (merge overlapping oblast- and raion-level rows per
   oblast) and local alerts (merge hromada-level rows per hromada, drop those
   that start during a regional alert of the same oblast).
4. Count alerts per day by start date in Kyiv time, measure the hours under
   alert per district for each day, then sum both by week (Monday to Sunday).
5. Download the eTryvoga log from the same repository and count its alerts per
   day in the same way. eTryvoga is a volunteer-run Telegram channel; its log
   has one row per alert per oblast and keeps running where the official log
   stops. It counts alerts differently from the official log (about 50% more
   since alerts moved to raion level), so it is kept as a separate dataset and
   never mixed into the official counts.

Outputs (``data/processed/``)
-----------------------------
daily_alerts.csv     date, alerts, regional_alerts, local_alerts,
                     oblasts_alerted (regions with at least one alert starting
                     that day, out of 25), alert_hours (hours under alert that
                     day, averaged over districts; 0 to 24), incomplete_data
weekly_alerts.csv    week_start (the Monday), alerts, regional_alerts,
                     local_alerts, alert_hours (0 to 168), days_complete (days
                     of the week fully covered by the log, out of 7),
                     incomplete_data
etryvoga_daily_alerts.csv
                     date, alerts, oblasts_alerted, incomplete_data, counted
                     from the eTryvoga log

Known limits of the source data
-------------------------------
* The official log starts on 15 March 2022.
* The permanent alerts in Luhansk oblast and Crimea are not in the log.
* The log starts and ends mid-day and can contain holes (periods with no
  records at all). Affected days and weeks are flagged ``incomplete_data``;
  their counts are too low, so exclude or impute them before modelling.
* A count of alerts says nothing about their length: where alerts run together
  into one long alert, as in front-line oblasts, the count falls. Use
  ``alert_hours`` when the question is how much time was spent under alert.

Usage
-----
    pip install pandas
    python data_getter.py                  # download (if missing) and process
    python data_getter.py --force-download # re-download the log first
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ALERTS_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/official_data_uk.csv"
)
ALERTS_FILE = "official_data_uk.csv"

ETRYVOGA_URL = (
    "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/"
    "main/datasets/volunteer_data_uk.csv"
)
ETRYVOGA_FILE = "volunteer_data_uk.csv"

KYIV_TZ_NAMES = ("Europe/Kyiv", "Europe/Kiev")  # the second is the pre-2022 tz name

# Regions without raions that count as one district when measuring time under alert.
SINGLE_DISTRICT_REGIONS = ["м. Київ"]

# The longest ordinary lull between two declarations anywhere in Ukraine is
# about 19 hours. A longer silence is treated as a hole in the log, and the
# days it touches are flagged as incomplete.
LOG_GAP_HOURS = 24


# --------------------------------------------------------------------------- #
# 1. Download
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it is already there."""
    if dest.exists() and not force:
        print(f"  using cached {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "data_getter.py"})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# --------------------------------------------------------------------------- #
# 2. Clean the alert log
# --------------------------------------------------------------------------- #
def load_alerts(alerts_path: Path) -> pd.DataFrame:
    """Read the log, drop duplicates and broken rows, parse timestamps as UTC."""
    a = pd.read_csv(alerts_path, dtype=str)
    n_raw = len(a)
    a = a.drop_duplicates(ignore_index=True)
    print(f"  {n_raw:,} rows in the log, {n_raw - len(a):,} exact duplicates dropped")

    a["start"] = pd.to_datetime(a["started_at"], utc=True, errors="coerce")
    a["end"] = pd.to_datetime(a["finished_at"], utc=True, errors="coerce")
    bad = a["start"].isna() | a["end"].isna() | (a["end"] < a["start"])
    if bad.any():
        print(f"  {int(bad.sum()):,} rows dropped: missing timestamp or end before start")
        a = a[~bad].reset_index(drop=True)

    # A few raion-wide alerts are logged at 'hromada' level with the raion name
    # repeated in the hromada column; treat them as the raion alerts they are.
    mislabelled = (a["level"] == "hromada") & (a["hromada"] == a["raion"])
    if mislabelled.any():
        print(f"  {int(mislabelled.sum()):,} rows relabelled from hromada to raion level")
        a.loc[mislabelled, "level"] = "raion"
        a.loc[mislabelled, "hromada"] = np.nan

    unknown = ~a["level"].isin(["oblast", "raion", "hromada"])
    if unknown.any():
        raise RuntimeError(f"unexpected values in 'level': {sorted(a.loc[unknown, 'level'].unique())}")
    return a[["oblast", "raion", "hromada", "level", "start", "end"]]


def load_etryvoga(path: Path) -> pd.DataFrame:
    """Read the eTryvoga log: one row per alert per oblast, timestamps in UTC."""
    v = pd.read_csv(path, dtype=str)
    n_raw = len(v)
    v = v.drop_duplicates(ignore_index=True)
    v["start"] = pd.to_datetime(v["started_at"], utc=True, errors="coerce")
    v["end"] = pd.to_datetime(v["finished_at"], utc=True, errors="coerce")
    bad = v["start"].isna() | v["end"].isna() | (v["end"] < v["start"])
    v = v[~bad].rename(columns={"region": "oblast"}).reset_index(drop=True)
    print(f"  {n_raw:,} rows in the log, {n_raw - len(v):,} duplicate or broken rows dropped")
    return v[["oblast", "start", "end"]]


# --------------------------------------------------------------------------- #
# 3. One row per alert
# --------------------------------------------------------------------------- #
def merge_overlaps(rows: pd.DataFrame, keys: list[str], gap_seconds: float = 0.0) -> pd.DataFrame:
    """Merge rows that share ``keys`` and whose time spans overlap or touch.

    Two rows are merged when the later one starts no more than ``gap_seconds``
    after the running end of the earlier ones. Returns one row per merged
    alert: the keys, ``start``, ``end`` and ``declarations`` (rows merged).
    """
    e = rows.sort_values(keys + ["start", "end"], ignore_index=True)
    unit = e.groupby(keys, sort=False).ngroup().to_numpy()
    # plain UTC nanoseconds, so the comparison does not depend on the pandas version
    start = e["start"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
    # latest end among this and all earlier rows of the same unit
    running_end = (e.groupby(keys, sort=False)["end"].cummax()
                   .dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64"))

    gap = int(round(gap_seconds * 1e9))
    new_alert = np.ones(len(e), dtype=bool)
    new_alert[1:] = (unit[1:] != unit[:-1]) | (start[1:] > running_end[:-1] + gap)

    # rows are sorted, so each merged alert is a consecutive block starting at a flagged row
    first = np.flatnonzero(new_alert)
    merged = e.loc[first, keys].reset_index(drop=True)
    merged["start"] = pd.to_datetime(start[first], utc=True)
    merged["end"] = pd.to_datetime(running_end[np.append(first[1:], len(e)) - 1], utc=True)
    merged["declarations"] = np.diff(np.append(first, len(e)))
    return merged


def build_alerts(log: pd.DataFrame, gap_seconds: float = 0.0) -> pd.DataFrame:
    """Turn the cleaned log into one row per alert: oblast, scope, start, end, declarations."""
    # Regional alerts: any oblast- or raion-level alert in force in the oblast.
    wide = log[log["level"].isin(["oblast", "raion"])]
    regional = merge_overlaps(wide, ["oblast"], gap_seconds)
    regional["scope"] = "regional"

    # Local alerts: single hromadas, merged per hromada.
    local = merge_overlaps(log[log["level"] == "hromada"], ["oblast", "raion", "hromada"], gap_seconds)

    # A local alert that starts while a regional alert is in force in its oblast
    # is part of that alert. Regional alerts of one oblast do not overlap, so it
    # is enough to look at the latest one that started before the local alert.
    latest = pd.merge_asof(
        local.sort_values("start"),
        regional[["oblast", "start", "end"]]
        .rename(columns={"start": "reg_start", "end": "reg_end"}).sort_values("reg_start"),
        left_on="start", right_on="reg_start", by="oblast", direction="backward",
    )
    covered = latest["reg_end"].notna() & (latest["start"] <= latest["reg_end"])
    local = latest.loc[~covered, ["oblast", "start", "end", "declarations"]].copy()
    local["scope"] = "local"
    print(f"  {len(regional):,} regional alerts; {len(local):,} local alerts "
          f"({int(covered.sum()):,} more were part of a regional alert)")

    columns = ["oblast", "scope", "start", "end", "declarations"]
    alerts = pd.concat([regional[columns], local[columns]], ignore_index=True)
    return alerts.sort_values(["start", "oblast"], ignore_index=True)


# --------------------------------------------------------------------------- #
# 4. Daily and weekly counts
# --------------------------------------------------------------------------- #
def to_kyiv_time(series: pd.Series) -> pd.Series:
    for name in KYIV_TZ_NAMES:
        try:
            return series.dt.tz_convert(name)
        except Exception:  # time-zone name unknown to this tz database
            continue
    raise RuntimeError("Kyiv time zone not found; install it with: pip install tzdata")


def kyiv_day(series: pd.Series) -> pd.Series:
    """Kyiv-time calendar day of each UTC timestamp."""
    return to_kyiv_time(series).dt.tz_localize(None).dt.normalize()


def find_log_gaps(declaration_starts: pd.Series) -> pd.DataFrame:
    """Periods longer than LOG_GAP_HOURS in which no alert was declared anywhere."""
    starts = declaration_starts.drop_duplicates().sort_values(ignore_index=True)
    silence = starts.diff()
    is_gap = silence > pd.Timedelta(hours=LOG_GAP_HOURS)
    return pd.DataFrame({
        "gap_start": starts.shift(1)[is_gap],
        "gap_end": starts[is_gap],
    }).reset_index(drop=True)


def daily_counts(alerts: pd.DataFrame, log_start: pd.Timestamp, log_end: pd.Timestamp,
                 log_gaps: pd.DataFrame) -> pd.DataFrame:
    """Number of alerts per day.

    An alert belongs to the Kyiv-time calendar day on which it started, so an
    alert that runs past midnight is counted once.
    """
    first_day, last_day = kyiv_day(pd.Series([log_start, log_end]))
    all_days = pd.date_range(first_day, last_day, freq="D", name="date")

    dated = alerts.assign(date=kyiv_day(alerts["start"]))
    by_day = dated.groupby("date")
    columns = {"alerts": by_day.size()}
    if "scope" in dated.columns:  # the official log: split the total into its two parts
        for scope in ("regional", "local"):
            columns[f"{scope}_alerts"] = dated[dated["scope"] == scope].groupby("date").size()
    columns["oblasts_alerted"] = by_day["oblast"].nunique()
    daily = pd.DataFrame(columns).reindex(all_days).fillna(0).astype(int).reset_index()

    # Days the log does not fully cover: it starts and ends mid-day, and it has holes.
    incomplete = pd.DatetimeIndex([first_day, last_day])
    for gap_first, gap_last in zip(kyiv_day(log_gaps["gap_start"]), kyiv_day(log_gaps["gap_end"])):
        incomplete = incomplete.append(pd.date_range(gap_first, gap_last, freq="D"))
    daily["incomplete_data"] = daily["date"].isin(incomplete)
    return daily


def daily_alert_hours(log: pd.DataFrame, days: pd.Series) -> pd.Series:
    """Hours under alert on each Kyiv-time day in ``days``, averaged over districts.

    A district (raion) is under alert while an alert is in force for the district
    itself or for its whole oblast. Alerts for single hromadas are left out.
    """
    districts = log.dropna(subset=["raion"])[["oblast", "raion"]].drop_duplicates()
    single = [region for region in SINGLE_DISTRICT_REGIONS if (log["oblast"] == region).any()]
    oblast_wide = log.loc[log["level"] == "oblast", ["oblast", "start", "end"]]
    spans = pd.concat([
        oblast_wide.merge(districts, on="oblast"),  # an oblast-wide alert covers each of its districts
        log.loc[log["level"] == "raion", ["oblast", "raion", "start", "end"]],
        oblast_wide[oblast_wide["oblast"].isin(single)].assign(raion=lambda frame: frame["oblast"]),
    ], ignore_index=True)
    under_alert = merge_overlaps(spans, ["oblast", "raion"])  # unbroken periods of alert per district
    n_districts = len(districts) + len(single)

    def to_ns(series: pd.Series) -> np.ndarray:
        return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]").astype("int64")

    # Kyiv midnights as UTC instants: day i runs from edges[i] to edges[i + 1]
    midnights = pd.Series(pd.date_range(days.iloc[0], days.iloc[-1] + pd.Timedelta(days=1), freq="D"))
    for name in KYIV_TZ_NAMES:
        try:
            edges = to_ns(midnights.dt.tz_localize(name))
            break
        except Exception:  # time-zone name unknown to this tz database
            continue
    else:
        raise RuntimeError("Kyiv time zone not found; install it with: pip install tzdata")

    start, end = to_ns(under_alert["start"]), to_ns(under_alert["end"])
    first = np.clip(np.searchsorted(edges, start, side="right") - 1, 0, len(days) - 1)
    last = np.clip(np.searchsorted(edges, end, side="left") - 1, 0, len(days) - 1)

    # an alert that runs past midnight contributes to every day it touches
    nanoseconds = np.zeros(len(days))
    for offset in range(int((last - first).max()) + 1):
        day = first + offset
        touches = day <= last
        overlap = np.minimum(end[touches], edges[day[touches] + 1]) - np.maximum(start[touches], edges[day[touches]])
        np.add.at(nanoseconds, day[touches], np.clip(overlap, 0, None))

    hours = nanoseconds / 3.6e12 / n_districts
    print(f"  time under alert measured over {n_districts} districts")
    return pd.Series(hours.round(3), index=days.index, name="alert_hours")


def weekly_counts(daily: pd.DataFrame) -> pd.DataFrame:
    """Sum the daily series by week, Monday to Sunday (Kyiv time).

    A week is flagged ``incomplete_data`` unless the log fully covers all seven
    of its days; its totals are then lower than the true figures.
    """
    week_start = daily["date"] - pd.to_timedelta(daily["date"].dt.dayofweek, unit="D")
    weekly = daily.assign(week_start=week_start).groupby("week_start").agg(
        alerts=("alerts", "sum"),
        regional_alerts=("regional_alerts", "sum"),
        local_alerts=("local_alerts", "sum"),
        alert_hours=("alert_hours", "sum"),
        days_complete=("incomplete_data", lambda flags: int((~flags).sum())),
    ).reset_index()
    weekly["alert_hours"] = weekly["alert_hours"].round(3)
    weekly["incomplete_data"] = weekly["days_complete"] < 7
    return weekly


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data",
                        help="folder for raw and processed data (default: ./data next to this script)")
    parser.add_argument("--force-download", action="store_true",
                        help="download the log again even if it is already there")
    parser.add_argument("--merge-gap-seconds", type=float, default=0.0,
                        help="also merge alerts separated by at most this many seconds (default: 0)")
    args = parser.parse_args(argv)

    # Ukrainian place names in messages must not crash consoles with a legacy code page
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    out_dir = args.data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1/5 Downloading the alert log")
    alerts_path = download(ALERTS_URL, args.data_dir / "raw" / ALERTS_FILE, args.force_download)

    print("2/5 Cleaning the log")
    log = load_alerts(alerts_path)
    log_start, log_end = log["start"].min(), log["end"].max()
    print(f"  log covers {log_start:%Y-%m-%d %H:%M} to {log_end:%Y-%m-%d %H:%M} UTC")
    log_gaps = find_log_gaps(log["start"])
    for gap in log_gaps.itertuples(index=False):
        print(f"  WARNING: hole in the log, no declarations from {gap.gap_start:%Y-%m-%d %H:%M} "
              f"to {gap.gap_end:%Y-%m-%d %H:%M} UTC")
    print("  declarations by level: "
          + ", ".join(f"{k} {v:,}" for k, v in log["level"].value_counts().items()))

    print("3/5 Building one row per alert")
    alerts = build_alerts(log, args.merge_gap_seconds)
    print(f"  {len(log):,} declarations -> {len(alerts):,} alerts in "
          f"{alerts['oblast'].nunique()} regions")

    print("4/5 Counting alerts and hours under alert per day (Kyiv time) and per week")
    daily = daily_counts(alerts, log_start, log_end, log_gaps)
    daily.insert(daily.columns.get_loc("incomplete_data"), "alert_hours", daily_alert_hours(log, daily["date"]))
    weekly = weekly_counts(daily)
    if int(daily["alerts"].sum()) != len(alerts) or int(weekly["alerts"].sum()) != len(alerts):
        raise RuntimeError("daily or weekly totals do not add up to the number of alerts")
    daily.to_csv(out_dir / "daily_alerts.csv", index=False, date_format="%Y-%m-%d")
    weekly.to_csv(out_dir / "weekly_alerts.csv", index=False, date_format="%Y-%m-%d")
    print(f"  {len(daily):,} days, {daily['date'].min():%Y-%m-%d} to {daily['date'].max():%Y-%m-%d}; "
          f"{int(daily['incomplete_data'].sum())} days flagged as incomplete")
    print(f"  {len(weekly)} weeks, starting {weekly['week_start'].min():%Y-%m-%d} to "
          f"{weekly['week_start'].max():%Y-%m-%d}; {int(weekly['incomplete_data'].sum())} flagged as incomplete")

    print("5/5 Counting alerts per day in the eTryvoga log")
    etryvoga_path = download(ETRYVOGA_URL, args.data_dir / "raw" / ETRYVOGA_FILE, args.force_download)
    etryvoga_log = load_etryvoga(etryvoga_path)
    # one row per alert per oblast already; merging only guards against overlapping rows
    etryvoga_alerts = merge_overlaps(etryvoga_log, ["oblast"], args.merge_gap_seconds)
    etryvoga_start, etryvoga_end = etryvoga_log["start"].min(), etryvoga_log["end"].max()
    etryvoga_gaps = find_log_gaps(etryvoga_log["start"])
    for gap in etryvoga_gaps.itertuples(index=False):
        print(f"  WARNING: hole in the eTryvoga log, no alerts from {gap.gap_start:%Y-%m-%d %H:%M} "
              f"to {gap.gap_end:%Y-%m-%d %H:%M} UTC")
    etryvoga_daily = daily_counts(etryvoga_alerts, etryvoga_start, etryvoga_end, etryvoga_gaps)
    if int(etryvoga_daily["alerts"].sum()) != len(etryvoga_alerts):
        raise RuntimeError("eTryvoga daily totals do not add up to the number of alerts")
    etryvoga_daily.to_csv(out_dir / "etryvoga_daily_alerts.csv", index=False, date_format="%Y-%m-%d")
    print(f"  {len(etryvoga_alerts):,} alerts; {len(etryvoga_daily):,} days, "
          f"{etryvoga_daily['date'].min():%Y-%m-%d} to {etryvoga_daily['date'].max():%Y-%m-%d}; "
          f"{int(etryvoga_daily['incomplete_data'].sum())} days flagged as incomplete")

    print(f"Done. Files are in {args.data_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
