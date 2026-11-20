"""Download the external DFL / IDSSE open matches used to train and calibrate the pipeline.

    python dead_ball/fetch_dfl.py

Source: Bassek, M., Rein, R., Weber, H., Memmert, D. "An integrated dataset of spatiotemporal and
event data in elite soccer." Scientific Data 12, 195 (2025), doi 10.1038/s41597-025-04505-y.
Data owner: Deutsche Fussball Liga (DFL). Licence: CC BY 4.0. Original hosting: Figshare,
doi 10.6084/m9.figshare.28196177. Downloaded here from the Hugging Face mirror pysport/idsse-data,
pinned to one revision so every run gets the same bytes; each file's size is checked.

Only the match information, the raw event files and the observed positions are fetched (~2.5 GB).
Files already present with the right size are skipped.

Output: $MTAH_DFL_RAW, by default MTAH_WORK_DIR/external/dfl_idsse/ (never commit it: the licence
allows redistribution, but it is large and not ours to re-host).
"""

import os
import sys
import urllib.request

# Params
WORK_DIR = os.environ.get("MTAH_WORK_DIR") or os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "work")
RAW = os.environ.get("MTAH_DFL_RAW") or os.path.join(WORK_DIR, "external", "dfl_idsse")
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}/{}"
SIZES = {
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml": 11930,
    "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml": 11722,
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml": 11745,
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml": 11952,
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml": 11922,
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml": 11932,
    "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml": 11823,
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml": 776578,
    "DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml": 612114,
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml": 635747,
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml": 683018,
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml": 698898,
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml": 732186,
    "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml": 681050,
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml": 418524833,
    "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml": 391722803,
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml": 348945715,
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml": 362567859,
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml": 371688499,
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml": 356219423,
    "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml": 372934140,
}


def main():
    os.makedirs(RAW, exist_ok=True)
    for name, size in SIZES.items():
        path = os.path.join(RAW, name)
        if os.path.exists(path) and os.path.getsize(path) == size:
            print("ok      ", name)
            continue
        print("download", name, "({:.0f} MB)".format(size / 1e6), flush=True)
        urllib.request.urlretrieve(URL.format(REVISION, name), path + ".part")
        if os.path.getsize(path + ".part") != size:
            sys.exit("size mismatch for {}: got {}, expected {}".format(name, os.path.getsize(path + ".part"), size))
        os.replace(path + ".part", path)
    print("all {} files present in {}".format(len(SIZES), RAW))


if __name__ == "__main__":
    main()
