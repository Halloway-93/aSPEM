import io
import json
import re
import warnings
from collections import defaultdict
from datetime import datetime
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import xarray as xr
from frites import set_mpl_style
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from scipy import stats
from scipy.stats import linregress, normaltest, pearsonr

set_mpl_style()

# |%%--%%| <LBiWLFEt1r|OOOEMsFiWW>

def process_events(rows, blocks, colnames):
    # If no data, create empty dataframe w/ all cols and types
    if len(rows) == 0:
        rows = ["", ""]
        blocks = []
    # Parse data, dropping useless first column
    if len(rows) == 1:
        list(rows).append("")
    # first col is event type, which we drop later
    colnames = ["type"] + colnames
    coltypes = get_coltypes(colnames)
    df = pd.read_csv(
        io.StringIO("\n".join(rows)),
        delimiter="\s+",
        header=None,
        names=colnames,
        na_values=".",
        index_col=False,
    )
    df = df.iloc[:, 1:]  # drop the first column
    # Move eye column to end & make factor, append block numbers to beginning of data frame
    if "eye" in colnames:
        df = df.iloc[:, [1] + list(range(2, df.shape[1])) + [0]]
        df["eye"] = pd.Categorical(df["eye"], categories=["L", "R"], ordered=False)
    df.insert(loc=0, column="trial", value=blocks)
    return df


def process_saccades(saccades, blocks, info):
    sacc_df = process_events(saccades, blocks, get_sacc_header(info))
    # Set amplitudes for any saccades missing start/end coords to NAs because they're wonky
    ampl_cols = [col for col in sacc_df.columns if re.search(r"ampl\d*$", col)]
    partial = sacc_df["sxp"].isna() | sacc_df["exp"].isna()
    if any(partial):
        sacc_df.loc[partial, ampl_cols] = pd.NA
    return sacc_df


def process_fixations(fixations, blocks, info):
    return process_events(fixations, blocks, get_fix_header(info))


def process_blinks(blinks, blocks):
    return process_events(blinks, blocks, ["eye", "stime", "etime", "dur"])


def process_messages(msgs, blocks):
    # Process messages from tracker
    msg_mat = [msg.split(" ", 1) for msg in msgs]
    msg_mat = [[msg[0][4:], msg[1]] for msg in msg_mat]
    msg_df = pd.DataFrame(msg_mat, columns=["time", "text"])
    msg_df["time"] = pd.to_numeric(msg_df["time"])

    # Append trial numbers to beginning of data frame
    msg_df.insert(0, "trial", blocks)

    return msg_df


def process_input(input_data, blocks):
    return process_events(input_data, blocks, ["time", "value"])


def process_buttons(button, blocks):
    return process_events(button, blocks, ["time", "button", "state"])

# |%%--%%| <OOOEMsFiWW|dklmovYrsV>

def from_header(header, field):
    pattern = r"\*\* {}\s*: (.*)".format(re.escape(field))
    matches = [re.findall(pattern, line) for line in header]
    matches = [match for match in matches if match]
    return matches[0][0] if matches else None


def get_resolution(nonsample):
    res = [None, None]
    for pattern in ["DISPLAY_COORDS", "GAZE_COORDS", "RESOLUTION"]:
        display_xy = [x for x in nonsample if pattern in x]
        if len(display_xy) == 0:
            continue
        display_xy = re.sub(f".* {pattern}\\D+(.*)", "\\1", display_xy[0])
        display_xy = [int(x) for x in re.split("\\s+", display_xy)]
        res = [display_xy[2] - display_xy[0] + 1, display_xy[3] - display_xy[1] + 1]
        break
    return res


def get_resolution(nonsample):
    res = [None, None]
    for pattern in ["DISPLAY_COORDS", "GAZE_COORDS", "RESOLUTION"]:
        display_xy = [s for s in nonsample if pattern in s]
        if len(display_xy) == 0:
            continue
        display_xy = re.sub(f".* {pattern}\\D+(.*)", "\\1", display_xy[0])
        try:
            display_xy = [int(float(s)) for s in display_xy.split()]
        except ValueError:
            continue
        res = [display_xy[2] - display_xy[0] + 1, display_xy[3] - display_xy[1] + 1]
        break
    return res

# |%%--%%| <dklmovYrsV|tb9YxS1CXf>

def get_mount(mount_str):
    # Older EyeLink 1000s may be missing "R" in table mount names, we add one if needed
    if re.search("TABLE$", mount_str):
        mount_str = mount_str + "R"

    mounts = {
        "MTABLER": "Desktop / Monocular / Head Stabilized",
        "BTABLER": "Desktop / Binocular / Head Stabilized",
        "RTABLER": "Desktop / Monocular / Remote",
        "RBTABLER": "Desktop / Binocular / Remote",
        "AMTABLER": "Arm Mount / Monocular / Head Stabilized",
        "ARTABLER": "Arm Mount / Monocular / Remote",
        "TOWER": "Tower Mount / Monocular / Head Stabilized",
        "BTOWER": "Tower Mount / Binocular / Head Stabilized",
        "MPRIM": "Primate Mount / Monocular / Head Stabilized",
        "BPRIM": "Primate Mount / Binocular / Head Stabilized",
        "MLRR": "Long-Range Mount / Monocular / Head Stabilized",
        "BLRR": "Long-Range Mount / Binocular / Head Stabilized",
    }

    return mounts[mount_str] if mount_str in mounts else None


def get_raw_header(info):
    eyev = ["xp", "yp", "ps"]

    if not info["mono"]:
        eyev = [f"{e}{s}" for s in ["l", "r"] for e in eyev]

    if info["velocity"]:
        if info["mono"]:
            eyev += ["xv", "yv"]
        else:
            eyev += [f"{e}{s}" for s in ["vl", "vr"] for e in ["x", "y"]]

    if info["resolution"]:
        eyev += ["xr", "yr"]

    if info["input"]:
        eyev += ["input"]

    if info["buttons"]:
        eyev += ["buttons"]

    if info["tracking"]:
        eyev += ["cr.info"]

    if info["htarg"]:
        eyev += ["tx", "ty", "td", "remote.info"]

    return ["time"] + eyev


def get_event_header(info, xy_cols):
    base = ["eye", "stime", "etime", "dur"]

    if info["event.dtype"] == "HREF":
        xy_cols = [f"href.{xy}" for xy in xy_cols] + xy_cols

    if info["resolution"]:
        xy_cols += ["xr", "yr"]

    return base + xy_cols


def get_sacc_header(info):
    return get_event_header(info, ["sxp", "syp", "exp", "eyp", "ampl", "pv"])


def get_fix_header(info):
    return get_event_header(info, ["axp", "ayp", "aps"])

# |%%--%%| <tb9YxS1CXf|QMbVIDuc2l>

def get_model(header):
    version_str = from_header(header, "VERSION")
    version_str2 = [x for x in header if re.search("\\*\\* EYELINK II", x)]
    if version_str is None:
        model = "Unknown"
        ver_num = "Unknown"
    elif version_str != "EYELINK II 1":
        model = "EyeLink I"
        ver_num = re.search(r"(\d+.\d+)", version_str).group(1)
    else:
        ver_num = re.search(r"v(\d+.\d+)", version_str2[0]).group(1)
        model = (
            "EyeLink II"
            if float(ver_num) < 2.4
            else "EyeLink 1000"
            if float(ver_num) < 5
            else "EyeLink 1000 Plus"
            if float(ver_num) < 6
            else "EyeLink Portable Duo"
        )
    return [model, ver_num]

# |%%--%%| <QMbVIDuc2l|XPme5pmfdX>

def get_coltypes(colnames, float_time=True):
    chr_cols = ["type", "eye", "cr.info", "remote.info"]
    int_cols = ["button", "state", "value"]
    time_cols = ["time", "stime", "etime", "dur"]
    if not float_time:
        int_cols += time_cols

    coltypes = [
        "str" if col in chr_cols else "int64" if col in int_cols else "float64"
        for col in colnames
    ]

    return coltypes


def get_htarg_regex(binocular):
    htarg_errs = "MANCFTBLRTBLRTBLR" if binocular else "MANCFTBLRTBLR"
    htarg_errs = list(htarg_errs)
    htarg_regex = "(" + "|".join(htarg_errs + ["\\."]) + ")"

    return htarg_regex


def is_float(string):
    return bool(re.search("\\.", string))

# |%%--%%| <XPme5pmfdX|QRvXrKqDhu>

def get_info(nonsample, firstcol):
    header = [f for f in nonsample if f.startswith("**")]
    info = {}

    # Get date/time of recording from file
    datetime.strptime(from_header(header, "DATE"), "%a %b %d %H:%M:%S %Y")
    # Get tracker model/version info
    version_info = get_model(header)
    info["model"] = version_info[0]
    info["version"] = version_info[1]

    # Get tracker mount info
    elclcfg = [line for line in nonsample if "ELCLCFG" in line]
    if len(elclcfg) > 0:
        info["mount"] = get_mount(re.findall(r"ELCLCFG\s+(.*)", elclcfg[0])[0])

    # Get display size from file
    screen_res = get_resolution(nonsample)
    info["screen.x"] = screen_res[0]
    info["screen.y"] = screen_res[1]

    # Get pupil size data type (area or diameter)
    pupil_config = [line for i, line in enumerate(nonsample) if firstcol[i] == "PUPIL"]
    if len(pupil_config) > 0:
        info["pupil.dtype"] = pupil_config[-1].split()[1]

    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "EVENTS"
    ]
    samples_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "SAMPLES"
    ]

    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "EVENTS"
    ]
    samples_config = [
        line for i, line in enumerate(nonsample) if firstcol[i] == "SAMPLES"
    ]
    if len(events_config) > 0:
        info["event.dtype"] = events_config[-1].split()[1]
    if len(samples_config) > 0:
        info["sample.dtype"] = samples_config[-1].split()[1]

    # Get last config line in file (preferring sample config) and extract remaining info
    config = events_config + samples_config[-1:]
    config = config[-1] if len(config) > 0 else ""
    if config:
        info["sample.rate"] = (
            float(re.findall(r"RATE\s+([0-9]+\.[0-9]+)", config)[0])
            if "RATE" in config
            else None
        )
        info["tracking"] = "\tTRACKING" in config
        info["cr"] = "\tCR" in config
        info["filter.level"] = (
            int(re.findall(r"FILTER\s+([0-9]+)", config)[0])
            if "FILTER" in config
            else None
        )
        info["velocity"] = "\tVEL" in config
        info["resolution"] = "\tRES" in config
        info["htarg"] = "\tHTARG" in config
        info["input"] = "\tINPUT" in config
        info["buttons"] = "\tBUTTONS" in config
        info["left"] = "\tLEFT" in config
        info["right"] = "\tRIGHT" in config
        info["mono"] = not (info["right"] & info["left"])

    return info

# |%%--%%| <QRvXrKqDhu|0kXKdf5eNc>

def process_raw(raw, blocks, info):
    if len(raw) == 0:
        # If no sample data in file, create empty raw DataFrame w/ all applicable columns
        raw = ["", ""]
        blocks = pd.Series([], dtype=int)
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time=False)
    else:
        # Determine if timestamps stored as floats (edf2asc option -ftime, useful for 2000 Hz)
        float_time = is_float(re.split(r"\s+", raw[0])[0])
        # Generate column names and types based in info in header
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time)
        # Discard any rows with too many or too few columns (usually rows where eye is missing)
        row_length = [len(re.split(r"\t", r)) for r in raw]
        med_length = np.median(row_length)
        raw = [r for r, l in zip(raw, row_length) if l == med_length]
        blocks = blocks[row_length == med_length]
        # Verify that generated columns match up with actual maximum row length
        length_diff = med_length - len(colnames)
        # if length_diff > 0:
        #    warnings.warn("Unknown columns in raw data. Assuming first one is time, please check the others")
        #    colnames = ["time"] + [f"X{i+1}" for i in range(med_length-1)]
        #    coltypes = "i" + "?"*(med_length-1)
    # Process raw sample data using pandas
    if len(raw) == 1:
        raw.append("")

    raw_df = pd.read_csv(
        io.StringIO("".join(raw)),
        sep="\t",
        header=None,
        names=colnames,
        na_values=np.nan,
        low_memory=False,
    )

    if info["tracking"] and not info["cr"]:
        # Drop CR column when not actually used
        raw_df = raw_df.drop(columns=["cr.info"])
    # Append block numbers to beginning of DataFrame
    raw_df.insert(0, "trial", blocks)
    # Replace missing pupil data (zeros) with NaNs
    if "X1" not in raw_df.columns:
        if info["mono"]:
            raw_df.loc[raw_df["ps"] == 0, "ps"] = np.nan
        else:
            raw_df.loc[raw_df["psl"] == 0, "psl"] = np.nan
            raw_df.loc[raw_df["psr"] == 0, "psr"] = np.nan
    return raw_df

# |%%--%%| <0kXKdf5eNc|miNeNEWwRI>

def read_asc(fname, samples=True, events=True, parse_all=False):
    with open(fname, "r") as f:
        inp = f.readlines()

    # Convert to ASCII
    inp = [line.encode("ascii", "ignore").decode() for line in inp]

    # Get strings prior to first tab for each line for faster string matching
    inp_first = [re.split(r"\s", s)[0] for s in inp]

    # Get the Trial info for each trial:
    bias = [
        s.split()[4] for s in inp if len(s.split()) > 4 and s.split()[2] == "Trialinfo:"
    ]
    direct = [
        s.split()[5] for s in inp if len(s.split()) > 4 and s.split()[2] == "Trialinfo:"
    ]
    # Check if any actual data recorded in file
    starts = [i for i, x in enumerate(inp_first) if x == "START"]
    if not starts:
        raise ValueError("No samples or events found in .asc file.")

    # Read metadata from file before processing
    is_raw = [bool(re.match("^[0-9]", line)) for line in inp_first]

    info = get_info(
        [line for line, raw in zip(inp, is_raw) if not raw],
        [first for first, raw in zip(inp_first, is_raw) if not raw],
    )

    # Do some extra processing/sanitizing if there's HTARG info in the file
    if info["htarg"]:
        inp, info = handle_htarg(inp, info, is_raw)

    # Find blocks and mark lines between block ENDs and next block STARTs
    dividers = starts + [len(inp)]
    block = np.cumsum([x == "START" for x in inp_first])
    block = block.astype(float)

    for i in range(1, len(dividers)):
        start = dividers[i - 1]
        end = dividers[i]
        endline = [j for j, x in enumerate(inp_first[start:end]) if x == "END"]
        if endline and endline[-1] < end - start:
            block[endline[0] + start : end] += 0.5

    # Unless parsing all input, drop any lines not within a block
    block[: dividers[0] + 1] += 0.5
    if not parse_all:
        in_block = np.floor(block) == block
        inp = [line for line, block_match in zip(inp, in_block) if block_match]
        inp_first = [
            first for first, block_match in zip(inp_first, in_block) if block_match
        ]
        is_raw = [raw for raw, block_match in zip(is_raw, in_block) if block_match]
        block = block[in_block]

    block = np.array(block)

    # Initialize dictionary of data output and process different data types
    out = {}
    if samples:
        out["raw"] = process_raw(
            [line for line, raw in zip(inp, is_raw) if raw], block[is_raw], info
        )
    if events:
        is_sacc = np.array(inp_first) == "ESACC"
        out["sacc"] = process_saccades(
            np.array(inp)[is_sacc], np.array(block)[is_sacc], info
        )

        is_fix = np.array(inp_first) == "EFIX"
        out["fix"] = process_fixations(
            np.array(inp)[is_fix], np.array(block)[is_fix], info
        )

        is_blink = np.array(inp_first) == "EBLINK"
        out["blinks"] = process_blinks(
            np.array(inp)[is_blink], np.array(block)[is_blink]
        )

        is_msg = np.array(inp_first) == "MSG"
        out["msg"] = process_messages(np.array(inp)[is_msg], np.array(block)[is_msg])

        is_input = np.array(inp_first) == "INPUT"
        out["input"] = process_input(np.array(inp)[is_input], np.array(block)[is_input])

        is_button = np.array(inp_first) == "BUTTON"
        out["button"] = process_buttons(
            np.array(inp)[is_button], np.array(block)[is_button]
        )

    # needed for parsing, but otherwise redundant with CR
    info["tracking"] = None

    out["info"] = info

    return out, np.array(bias, dtype="int"), np.array(direct, dtype="int") % 2

# |%%--%%| <miNeNEWwRI|2ADrAQIMn6>

path = "/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/"

categories = np.sort([f for f in listdir(path)])
namesCat = []
for cat in categories:
    namesCat.append(np.sort([f for f in listdir(path + cat)]))

# |%%--%%| <2ADrAQIMn6|MbFybW0qFo>

allFiles = []

for nameCat, cat in zip(namesCat, categories):
    filesCat = []
    for name in nameCat:
        files = np.sort([f for f in listdir(path + cat + "/" + name)])
        filesCat.append(files)
    allFiles.append(filesCat)

# |%%--%%| <MbFybW0qFo|qMpTQUdRRc>

allPaths = []
for cat, names, conditions in zip(categories, namesCat, allFiles):
    catPaths = []
    for name, namecond in zip(names, conditions):
        for condition in namecond:
            catPaths.append(path + cat + "/" + name + "/" + condition)
    allPaths.append(catPaths)

# |%%--%%| <qMpTQUdRRc|QZ2qKxu6BL>



# |%%--%%| <QZ2qKxu6BL|fRyHAcddd6>
r"""°°°
## Going through all the Kids
°°°"""
# |%%--%%| <fRyHAcddd6|aE3FKDvHVp>

meanPos = []
stdPos = []


meanVelo = []
stdVelo = []


switch = []

Probas = []
Directions = []

TS = []
SaccD = []

for f in allPaths[2]:
    data, bias, direct = read_asc(f)

    # Getting the probability from the name of the file
    proba = int(f[-6:-4]) / 100  # probability
    if proba == 0.0:
        proba = 1

    df = data["raw"]

    # Checking if the experiment was binorcular or monocular
    mono = data["info"]["mono"]
    # Putting all the data in the right format
    df["trial"] = pd.to_numeric(df["trial"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    if not mono:
        df["xpl"] = pd.to_numeric(df["xpl"], errors="coerce")
        df["ypl"] = pd.to_numeric(df["ypl"], errors="coerce")
        df["psl"] = pd.to_numeric(df["psl"], errors="coerce")
        df["xpr"] = pd.to_numeric(df["xpr"], errors="coerce")
        df["ypr"] = pd.to_numeric(df["ypr"], errors="coerce")
        df["psr"] = pd.to_numeric(df["psr"], errors="coerce")
    else:
        df["xp"] = pd.to_numeric(df["xp"], errors="coerce")
        df["ypl"] = pd.to_numeric(df["yp"], errors="coerce")
        df["ps"] = pd.to_numeric(df["ps"], errors="coerce")

    df["input"] = pd.to_numeric(df["input"], errors="coerce")

    # Messages from eyelink:
    MSG = data["msg"]
    t0 = MSG.loc[MSG.text == "StimulusOff\n", "time"]
    Zero = MSG.loc[MSG.text == "TargetOn\n", ["trial", "time"]]

    # resetting the time
    for i in range(len(Zero)):
        df.loc[df["trial"] == i + 1, "time"] = (
            df.loc[df["trial"] == i + 1, "time"] - Zero.time.values[i]
        )

    # Getting the saccades:
    Sacc = data["sacc"]

    # Resetting the saccades
    for t in Zero.trial:
        Sacc.loc[Sacc.trial == t, ["stime", "etime"]] = (
            Sacc.loc[Sacc.trial == t, ["stime", "etime"]].values
            - Zero.loc[Zero.trial == t, "time"].values
        )

    # Getting the trials where the saccades happends inside the time window. 0 and 80ms.
    trialSacc = Sacc[(Sacc.stime >= 0) & (Sacc.etime < 80) & (Sacc.eye == "R")][
        "trial"
    ].values

    saccDir = np.sign(
        (
            Sacc[(Sacc.stime >= 0) & (Sacc.etime < 80) & (Sacc.eye == "R")].exp
            - Sacc[(Sacc.stime >= 0) & (Sacc.etime < 80) & (Sacc.eye == "R")].sxp
        ).values
    )

    for t in Sacc.trial.unique():
        start = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "stime"]
        end = Sacc.loc[(Sacc.trial == t) & (Sacc.eye == "R"), "etime"]

        for i in range(len(start)):
            if not mono:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 20)
                    & (df.time <= end.iloc[i] + 20),
                    "xpr",
                ] = np.nan

            else:
                df.loc[
                    (df.trial == t)
                    & (df.time >= start.iloc[i] - 20)
                    & (df.time <= end.iloc[i] + 20),
                    "xp",
                ] = np.nan

    # first porbability switch
    first_bias = np.where(bias == 1)[0][0]
    switch.append(first_bias)

    if not mono:
        # Select the desired values
        selected_values = df.xpr[(df.time >= 80) & (df.time <= 120)]

        # print(len(selected_values))

        # Rescale the position:
        pos_before = df.xpr[(df.time >= -40) & (df.time <= 0)]

        # Reshape into a 2D matrix
        time_dim = 41

        trial_dim = len(selected_values) // time_dim

        # Re-basing the position
        pos = np.array(selected_values[: time_dim * trial_dim]).reshape(
            trial_dim, time_dim
        )
        # variance of pos on bias and non bias trials
        stdPos.append(np.std(pos, axis=1) / 30)
        pos_before_reshaped = np.array(pos_before[: time_dim * trial_dim]).reshape(
            trial_dim, time_dim
        )
        pos_before_mean = np.nanmean(pos_before_reshaped, axis=1)

        velo = np.gradient(pos, axis=1) * 1000 / 30
        velo[(velo > 30) | (velo < -30)] = np.nan

        for i, pp in enumerate(pos_before_mean):
            if pp == np.nan:
                pos[i] = np.nan
            else:
                pos[i] = (pos[i] - pp) / 30

        # pos=(pos-pos_before_mean.reshape(-1, 1))/30
        pos[(pos > 3) | (pos < -3)] = np.nan

    else:
        # Select the desired values
        selected_values = df.xp[(df.time >= 80) & (df.time <= 120)]
        # print(len(selected_values))

        # Rescale the position:
        pos_before = df.xp[(df.time >= -40) & (df.time <= 0)]

        # Reshape into a 2D matrix
        time_dim = 41

        trial_dim = len(selected_values) // time_dim

        pos = np.array(selected_values[: time_dim * trial_dim]).reshape(
            trial_dim, time_dim
        )
        stdPos.append(np.std(pos, axis=1) / 30)
        pos_before_reshaped = np.array(pos_before[: time_dim * trial_dim]).reshape(
            trial_dim, time_dim
        )
        pos_before_mean = np.nanmean(pos_before_reshaped, axis=1)

        velo = np.gradient(pos, axis=1) * 1000 / 30
        velo[(velo > 30) | (velo < -30)] = np.nan

        for i, pp in enumerate(pos_before_mean):
            if pp == np.nan:
                pos[i] = np.nan
            else:
                pos[i] = (pos[i] - pp) / 30

        # pos=(pos-pos_before_mean.reshape(-1, 1))/30
        pos[(pos > 3) | (pos < -3)] = np.nan

    # mean pos on bias and non bias trials for time window 80,120
    meanPos.append(np.mean(pos, axis=1))

    # mean of velocity on bias and non bias trials
    meanVelo.append(np.mean(velo, axis=1))

    # var of velocity on bias and non bias trials
    stdVelo.append(np.std(velo, axis=1))

    # subjects.append(name)

    Probas.append(proba)

    # Adding target directions

    Directions.append(direct)

    # Trials where there is a saccade in the time window [0, 80ms]

    TS.append(trialSacc)

    # Direction of the saccad

    SaccD.append(saccDir)

# |%%--%%| <aE3FKDvHVp|kOIJIpx1dB>

Sacc

# |%%--%%| <kOIJIpx1dB|weXSK2Ej72>

Sacc[(Sacc.stime > 0) & (Sacc.etime < 80) & (Sacc.eye == "R")]["dur"].values

# |%%--%%| <weXSK2Ej72|NtyyoHtUc0>

start.values

# |%%--%%| <NtyyoHtUc0|msEKz6vEyK>

end.values

# |%%--%%| <msEKz6vEyK|IAcdkQgnQ4>

df = pd.DataFrame(
    {
        "meanPos": meanPos,
        "stdPos": stdPos,
        "meanVelo": meanVelo,
        "stdVelo": stdVelo,
        "proba": Probas,
        "switch": switch,
        "tgt_direction": Directions,
        "SaccDirection": SaccD,
        "SaccTrial": TS,
    }
)
subjects = []
for name, file in zip(namesCat[2], allFiles[2]):
    for i in range(len(file)):
        subjects.append(name)
df["name"] = subjects

# |%%--%%| <IAcdkQgnQ4|NZ7BJuTlYi>

# Calculate the mean for 'meanPos' starting from the corresponding 'switch'
df["meanPosBias"] = df.apply(
    lambda row: pd.Series(row["meanPos"][row["switch"] :]).mean(), axis=1
)
df["meanPosNoBias"] = df.apply(
    lambda row: pd.Series(row["meanPos"][: row["switch"]]).mean(), axis=1
)
# Calculate the std for 'meanPos' starting from the corresponding 'switch'
df["stdPosBias"] = df.apply(
    lambda row: pd.Series(row["meanPos"][row["switch"] :]).std(), axis=1
)
df["stdPosNoBias"] = df.apply(
    lambda row: pd.Series(row["meanPos"][: row["switch"]]).std(), axis=1
)

# |%%--%%| <NZ7BJuTlYi|2M3oUTAXKv>

# Calculate the mean for 'meanPos' starting from theswitch
df["meanVeloBias"] = df.apply(
    lambda row: pd.Series(row["meanVelo"][row["switch"] :]).mean(), axis=1
)
df["meanVeloNoBias"] = df.apply(
    lambda row: pd.Series(row["meanVelo"][: row["switch"]]).mean(), axis=1
)

# Calculate the mean for 'stdPos' for the bias portion and nonbias portion
df["stdVeloBias"] = df.apply(
    lambda row: pd.Series(row["meanVelo"][row["switch"] :]).std(), axis=1
)
df["stdVeloNoBias"] = df.apply(
    lambda row: pd.Series(row["meanVelo"][: row["switch"]]).std(), axis=1
)

# |%%--%%| <2M3oUTAXKv|MH8SzADHHv>

# Identify columns with lists or NumPy arrays
list_columns = [col for col in df.columns if isinstance(df[col][0], (list, np.ndarray))]

# Serialize columns with lists/arrays to JSON strings
for col in list_columns:
    if isinstance(df[col][0], np.ndarray):
        df[col] = df[col].apply(lambda x: json.dumps(x.tolist()))
    else:
        df[col] = df[col].apply(json.dumps)

# |%%--%%| <MH8SzADHHv|vFnmVuF3Z6>

df.head()

# |%%--%%| <vFnmVuF3Z6|LBCP7ZWr82>

df.meanPos[0]

# |%%--%%| <LBCP7ZWr82|hMnIJW5VTj>

# Save the DataFrame to a CSV file
df.to_csv("Enfants.csv", index=False)

# |%%--%%| <hMnIJW5VTj|NhSAWevVaY>
r"""°°°
## No need to run the Function again 
°°°"""
# |%%--%%| <NhSAWevVaY|i6uCG38h5F>

# Read the CSV file
list_columns = [
    "meanPos",
    "stdPos",
    "meanVelo",
    "stdVelo",
    "tgt_direction",
    "SaccDirection",
    "SaccTrial",
]
df = pd.read_csv("Enfants.csv")

# |%%--%%| <i6uCG38h5F|5fkZvrQTLZ>

df.head()

df.meanPos[0]

# |%%--%%| <5fkZvrQTLZ|S8KgHmTcm2>

# Deserialize columns with lists/arrays back to their original data type
for col in list_columns:
    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# |%%--%%| <S8KgHmTcm2|yXzY73uzUP>

f.meanPos[0]

# |%%--%%| <yXzY73uzUP|cZ5aBe6Y93>



# |%%--%%| <cZ5aBe6Y93|2M8gXFkGQ7>

df[df["meanPos"].apply(lambda x: len(x)) == 150].name.values

# |%%--%%| <2M8gXFkGQ7|W0PjlVdcLw>

nonValidVeloName = df[df.meanVeloBias.isna()].name.values

# |%%--%%| <W0PjlVdcLw|MPpkgQWRd6>

nonValidVeloName

# |%%--%%| <MPpkgQWRd6|4ld5DS9XyY>

nonValidPosName = df[df.meanPosBias.isna()].name.values

# |%%--%%| <4ld5DS9XyY|hgHsZTEOpt>

nonValidPosName

# |%%--%%| <hgHsZTEOpt|UrFgI6ZbbJ>

p_values = []
pbs = []
for i, s in enumerate(df.switch.values):
    vv = np.array(df.iloc[i].meanVelo[s:])
    if len(vv) > 0:
        vv = vv[~np.isnan(vv)]
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vv)
        # print("Bias test statistic:", jarque_bera_stat)
        print("Bias p-value:", jarque_bera_p_value)
        p_values.append(jarque_bera_p_value)
        pbs.append(df.iloc[i].proba)
        plt.hist(vv, label="Velo Bias", alpha=0.5)

    # Non Bias block
    vvv = np.array(df.iloc[i].meanVelo[:s])
    if len(vvv) > 0:
        vvv = vvv[~np.isnan(vvv)]
        plt.hist(vvv, label="Velo no Bias", alpha=0.5)
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vvv)

        # print("No Bias test statistic:", jarque_bera_stat)
        print("No Bias p-value:", jarque_bera_p_value)
    plt.title(df.iloc[i]["name"] + ", " + str(df.iloc[i].proba))
    plt.legend()
    plt.show()

# |%%--%%| <UrFgI6ZbbJ|PxSTVYvLdc>

PVS = pd.DataFrame({"pvalue": p_values, "proba": pbs})
sns.boxplot(data=PVS, x="proba", y="pvalue")

# |%%--%%| <PxSTVYvLdc|kLhEqY2HxK>

len(PVS[(PVS.pvalue < 0.01) & (PVS.proba != 0.5)])

# |%%--%%| <kLhEqY2HxK|zxVKaZxPSO>

sns.histplot(data=PVS, x="pvalue", hue="proba")

# |%%--%%| <zxVKaZxPSO|aBRHBGY8Tx>

p_values = []
pbs = []
for i, s in enumerate(df.switch.values):
    vv = np.array(df.iloc[i].meanPos[s:])
    if len(vv) > 0:
        vv = vv[~np.isnan(vv)]
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vv)
        # print("Bias test statistic:", jarque_bera_stat)
        print("Bias p-value:", jarque_bera_p_value)
        p_values.append(jarque_bera_p_value)
        pbs.append(df.iloc[i].proba)
        plt.hist(vv, label="Pos Bias", alpha=0.5)

    # Non Bias block
    vvv = np.array(df.iloc[i].meanPos[:s])
    if len(vvv) > 0:
        vvv = vvv[~np.isnan(vvv)]
        plt.hist(vvv, label="Pos No Bias", alpha=0.5)
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vvv)

        # print("No Bias test statistic:", jarque_bera_stat)
        print("No Bias p-value:", jarque_bera_p_value)
    plt.title(df.iloc[i]["name"] + ", " + str(df.iloc[i].proba))
    plt.legend()
    plt.show()

# |%%--%%| <aBRHBGY8Tx|Z1fEyCrsfj>

PVS = pd.DataFrame({"pvalue": p_values, "proba": pbs})
sns.boxplot(data=PVS, x="proba", y="pvalue")

# |%%--%%| <Z1fEyCrsfj|tTcXEU55zc>

len(PVS[(PVS.pvalue < 0.01)])

# |%%--%%| <tTcXEU55zc|ktJei66Tlk>

df.tgt_direction

# |%%--%%| <ktJei66Tlk|e5cxOeyJ7B>

l = (
    df[
        [
            "proba",
            "name",
            "meanVeloBias",
            "meanVeloNoBias",
            "stdVeloBias",
            "stdVeloNoBias",
            "meanPosBias",
            "meanPosNoBias",
            "stdPosBias",
            "stdPosNoBias",
        ]
    ]
    .groupby(["name", "proba"])
    .mean()
)
l.reset_index(inplace=True)
l

# |%%--%%| <e5cxOeyJ7B|xPzUgVPzhT>

sns.lmplot(data=l, x="proba", y="meanVeloBias")

# |%%--%%| <xPzUgVPzhT|PNLdHCvXhK>

sns.lmplot(data=l, x="meanPosBias", y="stdPosBias", hue="proba", height=10)

# |%%--%%| <PNLdHCvXhK|KDcOthuZ52>

sns.lmplot(data=l, x="meanVeloBias", y="stdVeloBias", hue="proba", height=10)
plt.savefig("corrVeloBiasEnfants")

# |%%--%%| <KDcOthuZ52|ZvKyLxGmJV>

sns.lmplot(data=l, x="meanPosBias", y="meanVeloBias", hue="proba", height=10)

# |%%--%%| <ZvKyLxGmJV|ao8zoBstI5>

Slope = []
for n in np.unique(df.name):
    slope, intercept, r_value, p_value, std_err = linregress(
        df[df.name == n].proba, df[df.name == n].meanPosBias
    )
    plt.figure(figsize=(12, 6))
    plt.scatter(df[df.name == n].proba, df[df.name == n].meanPosBias)
    plt.title(n + ", slope=" + str("% 6.3f" % slope) + " p=" + str("% 6.3f" % p_value))
    plt.xlabel("Proba")
    plt.ylabel("meanPosBias Bias")

    # Getting the slope
    Slope.append(slope)

    plt.plot(
        df[df.name == n].proba,
        df[df.name == n].proba * slope + intercept,
        c="r",
        alpha=0.5,
    )
    plt.show()

# |%%--%%| <ao8zoBstI5|Bj9NcRNBBQ>

ss = np.array(Slope)
Slope = ss[~np.isnan(ss)]
s = l.loc[~l.name.isin(nonValidPosName)].groupby("name").mean().stdPosNoBias.values

# |%%--%%| <Bj9NcRNBBQ|xje4IZeWQv>

plt.scatter(s, Slope)
plt.xlabel("stdPosNoBias")
plt.ylabel("Slope")

# |%%--%%| <xje4IZeWQv|VzIoLaf4Sc>

Slope = []
for n in np.unique(df.name):
    slope, intercept, r_value, p_value, std_err = linregress(
        df[df.name == n].proba, df[df.name == n].meanVeloBias
    )
    plt.figure(figsize=(12, 6))
    plt.scatter(df[df.name == n].proba, df[df.name == n].meanVeloBias)
    plt.title(n + ", slope=" + str("% 6.3f" % slope) + " p=" + str("% 6.3f" % p_value))
    plt.xlabel("Proba")
    plt.ylabel("meanVelo Bias")

    # Getting the slope
    Slope.append(slope)

    plt.plot(
        df[df.name == n].proba,
        df[df.name == n].proba * slope + intercept,
        c="r",
        alpha=0.5,
    )
    plt.show()

# |%%--%%| <VzIoLaf4Sc|J9Il9RR6yr>

ss = np.array(Slope)

# |%%--%%| <J9Il9RR6yr|MWwePnaIR5>

Slope = ss[~np.isnan(ss)]

# |%%--%%| <MWwePnaIR5|55FEzts5Hm>

s = l.loc[~l.name.isin(nonValidVeloName)].groupby("name").mean().stdVeloNoBias.values

# |%%--%%| <55FEzts5Hm|q0JaMPZeNh>

len(s)

# |%%--%%| <q0JaMPZeNh|oJjXhOV7ZW>

plt.scatter(s, Slope)
plt.xlabel("stdVeloNoBias")
plt.ylabel("Slope")

# |%%--%%| <oJjXhOV7ZW|Srm9dX4fAM>

data = {
    "X": s,
    "Y": Slope,
}

# Create a DataFrame
relBiasNoBias = pd.DataFrame(data)

# Add a constant (intercept) to the independent variable
X = sm.add_constant(relBiasNoBias["X"])

# Fit the linear regression model
model = sm.OLS(relBiasNoBias["Y"], X).fit()

# Get the R-squared (R²) value
r_squared = model.rsquared
print(f"R-squared (R²) value: {r_squared:.4f}")

# Get the p-value for the slope coefficient (X)
p_value = model.pvalues["X"]
print(f"P-value for the slope coefficient: {p_value:.4f}")

# Print the regression summary
print(model.summary())

# |%%--%%| <Srm9dX4fAM|MQP8kRCLlM>

# Get the regression results
results = model.get_robustcov_results(cov_type="HC3")

# Create a scatterplot
sns.scatterplot(x="X", y="Y", data=relBiasNoBias)

# Plot the regression line
sns.lineplot(
    x=relBiasNoBias["X"], y=model.fittedvalues, color="red", label="Regression Line"
)
X_values = relBiasNoBias["X"]
confidence_interval = results.conf_int()
upper_bound = confidence_interval[:, 1]  # Access the upper bound
lower_bound = confidence_interval[:, 0]  # Access the lower bound
# plt.fill_between(X_values, lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
plt.xlabel("stdVeloNoBias")
plt.ylabel("Slope")
plt.savefig("SlopeEnfants")

# |%%--%%| <MQP8kRCLlM|0twU84nGcE>

sns.boxplot(data=df, x="proba", y="meanPosBias")

# |%%--%%| <0twU84nGcE|MoVVTwqp1v>

sns.boxplot(data=df, x="proba", y="stdPosBias")

# |%%--%%| <MoVVTwqp1v|UGm32pbmoh>

sns.boxplot(data=df, x="proba", y="meanPosNoBias")

# |%%--%%| <UGm32pbmoh|zrhnyxS4mC>

sns.scatterplot(data=df, x="stdPosBias", y="meanPosBias", s=100, hue="proba")

# |%%--%%| <zrhnyxS4mC|Slo9P43hOd>

sns.scatterplot(
    data=df[df.proba == 1], x="stdPosBias", y="meanPosBias", s=100, hue="proba"
)

# |%%--%%| <Slo9P43hOd|D5Wd8M4Awf>

sns.scatterplot(
    data=df[~df.name.isin(nonValidVeloName)],
    x="meanVeloBias",
    y="meanPosBias",
    s=100,
    hue="proba",
)
plt.title("Correlation between the means of Position & Velocity in [80ms,120ms]")
plt.savefig("CorrPosVeloBias")

# |%%--%%| <D5Wd8M4Awf|nT0iq6rwPn>

slope, intercept, r_value, p_value, std_err = linregress(
    df[~df.name.isin(nonValidVeloName)].meanVeloBias,
    df[~df.name.isin(nonValidPosName)].meanPosBias,
)
plt.figure(figsize=(12, 6))
plt.scatter(df.meanVeloBias, df.meanPosBias)
plt.title(
    "Correlation between Velo and Pos Bias block R="
    + str("% 6.3f" % r_value)
    + " p="
    + str("% 6.3f" % p_value)
)
plt.xlabel("Velo")
plt.ylabel("Pos")

plt.plot(df.meanVeloBias, df.meanVeloBias * slope + intercept, c="r", alpha=0.5)

# |%%--%%| <nT0iq6rwPn|sfbskc4BN9>

slope, intercept, r_value, p_value, std_err = linregress(
    df[~df.name.isin(nonValidVeloName)].stdPosBias,
    df[~df.name.isin(nonValidVeloName)].meanPosBias,
)
plt.figure(figsize=(12, 6))
plt.scatter(df.stdPosBias, df.meanPosBias)
plt.title(
    "Correlation between StdPos and meanPos No Bias block R="
    + str("% 6.3f" % r_value)
    + " p="
    + str("% 6.3f" % p_value)
)
plt.xlabel("StdPos")
plt.ylabel("meanPos")

plt.plot(df.stdPosBias, df.stdPosBias * slope + intercept, c="r", alpha=0.5)

# |%%--%%| <sfbskc4BN9|WSvYWNwkVj>

slope, intercept, r_value, p_value, std_err = linregress(
    df[~df.name.isin(nonValidVeloName)].stdVeloBias,
    df[~df.name.isin(nonValidVeloName)].meanVeloBias,
)
plt.figure(figsize=(12, 6))
plt.scatter(df.stdVeloBias, df.meanVeloBias)
plt.title(
    "Correlation between StdVelo and meanVelo Bias block R="
    + str("% 6.3f" % r_value)
    + " p="
    + str("% 6.3f" % p_value)
)
plt.xlabel("StdVelo")
plt.ylabel("meanVelo")

plt.plot(df.stdVeloBias, df.stdVeloBias * slope + intercept, c="r", alpha=0.5)

# |%%--%%| <WSvYWNwkVj|AVhpTDD1RN>

sns.lmplot(
    data=df[~df.name.isin(nonValidVeloName)],
    x="meanVeloBias",
    y="stdVeloBias",
    height=10,
    hue="proba",
)
plt.title("Correlation between StdVelo and meanVelo Bias block")
plt.savefig("CorrStdMeanVeloBiasEnfants")

# |%%--%%| <AVhpTDD1RN|XFLrczTIDU>

sns.scatterplot(
    data=df[~df.name.isin(nonValidVeloName)],
    x="stdPosNoBias",
    y="meanPosNoBias",
    s=100,
    hue="proba",
)

# |%%--%%| <XFLrczTIDU|aQchmTPQCd>

sns.scatterplot(
    data=df[~df.name.isin(nonValidVeloName)],
    x="stdVeloNoBias",
    y="meanVeloNoBias",
    s=100,
    hue="proba",
)

# |%%--%%| <aQchmTPQCd|G7ArHOpCMZ>

# sns.scatterplot(data=df,x='CVVeloBias' ,y='meanVeloBias',s=100,hue='proba')

# |%%--%%| <G7ArHOpCMZ|aVywSe5YeV>

sns.scatterplot(
    data=df[(df.proba == 0.7) & (~df.name.isin(nonValidVeloName))],
    x="stdVeloBias",
    y="meanVeloBias",
    s=100,
    hue="proba",
)

# |%%--%%| <aVywSe5YeV|B656VgZVsa>

sns.scatterplot(
    data=df[(df.proba == 0.9) & (~df.name.isin(nonValidVeloName))],
    x="stdVeloBias",
    y="meanVeloBias",
    s=100,
    hue="proba",
)

# |%%--%%| <B656VgZVsa|xlSwLa17C7>

sns.scatterplot(
    data=df[(df.proba == 1) & (~df.name.isin(nonValidVeloName))],
    x="stdVeloBias",
    y="meanVeloBias",
    s=100,
    hue="proba",
)

# |%%--%%| <xlSwLa17C7|FoM0EiPp4S>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot your data or customize each subplot as needed
axs[0, 0].scatter(
    df[df.proba == 0.7]["meanPosBias"], df[df.proba == 0.7]["stdPosBias"], label="Bias"
)
axs[0, 0].scatter(
    df[df.proba == 0.7]["meanPosNoBias"],
    df[df.proba == 0.7]["stdPosNoBias"],
    label="No Bias",
)
axs[0, 0].set_ylabel("Proba=0.7")
axs[0, 0].set_xlabel("meanPosBias")
axs[0, 0].set_title("Position", pad=20)
axs[0, 1].scatter(
    df[df.proba == 0.7].meanVeloBias, df[df.proba == 0.7].stdVeloBias, label="Bias"
)
axs[0, 1].scatter(
    df[df.proba == 0.7].meanVeloNoBias,
    df[df.proba == 0.7].stdVeloNoBias,
    label=" NoBias",
)
axs[0, 1].set_title("Velocity", pad=20)
axs[0, 1].set_xlabel("meanVeloBias")
axs[0, 1].set_ylabel("stdVeloBias")
axs[1, 0].scatter(
    df[df.proba == 0.9].meanPosBias, df[df.proba == 0.9].stdPosBias, label="Bias"
)
axs[1, 0].scatter(
    df[df.proba == 0.9].meanPosNoBias, df[df.proba == 0.9].stdPosNoBias, label="No Bias"
)

axs[1, 0].set_ylabel("Proba=0.9")
axs[1, 1].scatter(
    df[df.proba == 0.9].meanVeloBias, df[df.proba == 0.9].stdVeloBias, label="Bias"
)
axs[1, 1].scatter(
    df[df.proba == 0.9].meanVeloNoBias,
    df[df.proba == 0.9].stdVeloNoBias,
    label="No Bias",
)
axs[1, 1].set_ylabel("stdVeloBias")


axs[2, 0].scatter(
    df[df.proba == 1].meanPosBias, df[df.proba == 1].stdPosBias, label="Bias"
)
axs[2, 0].scatter(
    df[df.proba == 1].meanPosNoBias, df[df.proba == 1].stdPosNoBias, label="No Bias"
)
axs[2, 0].set_ylabel("Proba=1")

axs[2, 1].scatter(
    df[df.proba == 1].meanVeloBias, df[df.proba == 1].stdVeloBias, label="Bias"
)
axs[2, 1].scatter(
    df[df.proba == 1].meanVeloNoBias, df[df.proba == 1].stdVeloNoBias, label="No Bias"
)
fig.suptitle(
    "Correlation between the mean and the standard deviation\n of Velocity & Position for different Probabilities ",
    fontsize=20,
)

# Add legends to each subplot
for ax in axs.flatten():
    ax.legend()

fig.tight_layout()
plt.savefig("CorrstdMeanProbasEnfants")

# |%%--%%| <FoM0EiPp4S|TWpm3BbiSV>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot your data or customize each subplot as needed
axs[0, 0].scatter(df[df.proba == 0.7]["meanPosBias"], df[df.proba == 0.7]["stdPosBias"])
axs[0, 0].set_ylabel("Proba=0.7")
axs[0, 0].set_xlabel("meanPosBias")
axs[0, 0].set_title("Position", pad=20)
axs[0, 1].scatter(df[df.proba == 0.7].meanVeloBias, df[df.proba == 0.7].stdVeloBias)
axs[0, 1].set_title("Velocity", pad=20)
axs[0, 1].set_xlabel("meanVeloBias")
axs[0, 1].set_ylabel("stdVeloBias")
axs[1, 0].scatter(df[df.proba == 0.9].meanPosBias, df[df.proba == 0.9].stdPosBias)
axs[1, 0].set_ylabel("Proba=0.9")
axs[1, 1].scatter(df[df.proba == 0.9].meanVeloBias, df[df.proba == 0.9].stdVeloBias)
axs[2, 0].scatter(df[df.proba == 1].meanPosBias, df[df.proba == 1].stdPosBias)
axs[2, 0].set_ylabel("Proba=1")
axs[2, 1].scatter(df[df.proba == 1].meanVeloBias, df[df.proba == 1].stdVeloBias)
fig.suptitle("Correlation between the mean and the standard deviation", fontsize=20)
fig.tight_layout()

# |%%--%%| <TWpm3BbiSV|V0zVsXmxMY>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# put the hue for the subjects
# Plot your data or customize each subplot as needed
slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[0, 0].scatter(
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[0, 0].plot(
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanPosBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[0, 0].set_ylabel("Proba=0.7")
axs[0, 0].set_xlabel("meanPosBias")
axs[0, 0].set_title("Position" + " p=" + str("% 6.3f" % p_value), pad=20)

slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 0.7) & (df.stdVeloBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.7) & (df.stdVeloBias <= 20)]["stdVeloBias"],
)
axs[0, 1].scatter(
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["stdVeloBias"],
)
axs[0, 1].plot(
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.7) & (df.stdPosBias <= 20)]["meanVeloBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[0, 1].set_ylabel("stdVeloBias")
axs[0, 1].set_xlabel("meanVeloBias")
axs[0, 1].set_title("Velo" + " p=" + str("% 6.3f" % p_value), pad=20)

slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[1, 0].scatter(
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[1, 0].plot(
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanPosBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[1, 0].set_ylabel("Proba=0.9")
axs[1, 0].set_xlabel("meanPosBias")
axs[1, 0].set_title("Position" + " p=" + str("% 6.3f" % p_value), pad=20)

slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 0.9) & (df.stdVeloBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.9) & (df.stdVeloBias <= 20)]["stdVeloBias"],
)
axs[1, 1].scatter(
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["stdVeloBias"],
)
axs[1, 1].plot(
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 0.9) & (df.stdPosBias <= 20)]["meanVeloBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[1, 1].set_ylabel("stdVeloBias")
axs[1, 1].set_xlabel("meanVeloBias")
axs[1, 1].set_title("Velo" + " p=" + str("% 6.3f" % p_value), pad=20)


slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[2, 0].scatter(
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["stdPosBias"],
)
axs[2, 0].plot(
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanPosBias"],
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanPosBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[2, 0].set_ylabel("Proba=1")
axs[2, 0].set_xlabel("meanPosBias")
axs[2, 0].set_title("Position" + " p=" + str("% 6.3f" % p_value), pad=20)

slope, intercept, r_value, p_value, std_err = linregress(
    df[(df.proba == 1) & (df.stdVeloBias <= 20)]["meanVeloBias"],
    df[(df.proba == 1) & (df.stdVeloBias <= 20)]["stdVeloBias"],
)
axs[2, 1].scatter(
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["stdVeloBias"],
)
axs[2, 1].plot(
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanVeloBias"],
    df[(df.proba == 1) & (df.stdPosBias <= 20)]["meanVeloBias"] * slope + intercept,
    c="r",
    alpha=0.5,
)
axs[2, 1].set_ylabel("stdVeloBias")
axs[2, 1].set_xlabel("meanVeloBias")
axs[2, 1].set_title("Velo" + " p=" + str("% 6.3f" % p_value), pad=20)


fig.suptitle(
    "Correlation between the mean and the standard deviation\n Bias Block  ",
    fontsize=20,
)
fig.tight_layout()
plt.savefig("CorrstdMeanProbasEnfantsBias")

# |%%--%%| <V0zVsXmxMY|AKVP6rocGK>

sns.boxplot(data=df, x="proba", y="meanVeloBias")

# |%%--%%| <AKVP6rocGK|cCVnLNPqe0>

sns.boxplot(data=df, x="proba", y="stdVeloBias")

# |%%--%%| <cCVnLNPqe0|i11XKSY1Ws>

sns.boxplot(data=df, x="proba", y="stdPosBias")

# |%%--%%| <i11XKSY1Ws|3eYAEdH4YT>

sns.boxplot(data=df, x="proba", y="stdPosNoBias")

# |%%--%%| <3eYAEdH4YT|mrNl8I9GEQ>

sns.boxplot(data=df, x="proba", y="stdVeloBias")

# |%%--%%| <mrNl8I9GEQ|LgSE9DL2a1>

sns.boxplot(data=df, x="proba", y="stdVeloNoBias")

# |%%--%%| <LgSE9DL2a1|q2wrdWyFVV>
r"""°°°
## Does the saccade comes to compensate in the opposite motion of anticipation
°°°"""
# |%%--%%| <q2wrdWyFVV|TPjMUoGa5K>

df.head()

# |%%--%%| <TPjMUoGa5K|L4x2I3V0oj>

df_filtered = df[(df["SaccDirection"].apply(len) > 0) & (df.proba == 1)][
    ["meanPos", "meanVelo", "SaccDirection", "SaccTrial", "tgt_direction", "switch"]
]
# df_filtered.head()
# df_filtered

# |%%--%%| <L4x2I3V0oj|1841MmmHhq>

def filter_values_bias(row):
    return [val for val in row["SaccTrial"] if val >= row["switch"]]


def filter_values_non_bias(row):
    return [val for val in row["SaccTrial"] if val < row["switch"]]

# |%%--%%| <1841MmmHhq|8Hfk9TtKK1>

df_filtered["SaccTrialBias"] = df_filtered.apply(filter_values_bias, axis=1)
df_filtered["SaccTrialNoBias"] = df_filtered.apply(filter_values_non_bias, axis=1)
df_filtered.SaccTrialNoBias.values[0]

# |%%--%%| <8Hfk9TtKK1|9fptYFngNr>

# df_filtered.meanVelo.iloc[0]
df_filtered["nBias"] = df_filtered.SaccTrialBias.apply(len).values
df_filtered["nNonBias"] = df_filtered.SaccTrialNoBias.apply(len).values
df_filtered["SaccDirectionBias"] = df_filtered.apply(
    lambda row: row["SaccDirection"][row["nNonBias"] :], axis=1
)
df_filtered["SaccDirectionNoBias"] = df_filtered.apply(
    lambda row: row["SaccDirection"][: row["nNonBias"]], axis=1
)
df_filtered.nNonBias.values[0]

# |%%--%%| <9fptYFngNr|LZSxFSgbJs>

# Uncomment this to see the values
# len(df_filtered.SaccDirectionBias.values[0])
# [len(df_filtered.SaccDirectionNoBias.values[i])for i in range(len(df_filtered))]

# |%%--%%| <LZSxFSgbJs|RgVRFbKA0c>

# [len(df_filtered.SaccTrialNoBias.values[i])for i in range(len(df_filtered))]==[len(df_filtered.SaccDirectionNoBias.values[i])for i in range(len(df_filtered))]

# |%%--%%| <RgVRFbKA0c|rdSax2D8ZX>

def extract_values(row, col):
    sacc_trial_indices = [int(idx) - 1 for idx in row[col]]

    meanVelo_values = []
    tgt_direction_values = []
    meanPosValues = []
    # Iterate through the indices and append the values
    for idx in sacc_trial_indices:
        # print(len(row['meanVelo']))
        if 0 <= idx < len(row["meanVelo"]):
            meanVelo_values.append(row["meanVelo"][idx])
            tgt_direction_values.append(row["tgt_direction"][idx])
            meanPosValues.append(row["meanPos"][idx])
        else:
            print(f"Invalid index {idx} for meanVelo")
    return meanPosValues, meanVelo_values, tgt_direction_values

# |%%--%%| <rdSax2D8ZX|wtXKsceXnX>

result = df_filtered.apply(lambda row: extract_values(row, "SaccTrial"), axis=1)
resultBias = df_filtered.apply(lambda row: extract_values(row, "SaccTrialBias"), axis=1)
resultNoBias = df_filtered.apply(
    lambda row: extract_values(row, "SaccTrialNoBias"), axis=1
)

# |%%--%%| <wtXKsceXnX|wjml37vHj8>

# df_filtered.SaccDirection.values

# |%%--%%| <wjml37vHj8|oRA2NFGVKS>

resultNoBias

# |%%--%%| <oRA2NFGVKS|yu0KCz7WW9>

POS = [
    value
    for sublist in [result.iloc[i][0] for i in range(len(result))]
    for value in sublist
]
MV = [
    value
    for sublist in [result.iloc[i][1] for i in range(len(result))]
    for value in sublist
]
TD = [
    value
    for sublist in [result.iloc[i][2] for i in range(len(result))]
    for value in sublist
]
# Extract and flatten the list of meanVelo_values
POSB = [
    value
    for sublist in [resultBias.iloc[i][0] for i in range(len(resultBias))]
    for value in sublist
]
MVB = [
    value
    for sublist in [resultBias.iloc[i][1] for i in range(len(resultBias))]
    for value in sublist
]
TDB = [
    value
    for sublist in [resultBias.iloc[i][2] for i in range(len(resultBias))]
    for value in sublist
]
POSNB = [
    value
    for sublist in [resultNoBias.iloc[i][0] for i in range(len(resultNoBias))]
    for value in sublist
]
MVNB = [
    value
    for sublist in [resultNoBias.iloc[i][1] for i in range(len(resultNoBias))]
    for value in sublist
]
TDNB = [
    value
    for sublist in [resultNoBias.iloc[i][2] for i in range(len(resultNoBias))]
    for value in sublist
]

# |%%--%%| <yu0KCz7WW9|mX2Itr1fGW>

SD = [value for sublist in df_filtered.SaccDirection.values for value in sublist]
SDB = [value for sublist in df_filtered.SaccDirectionBias.values for value in sublist]
SDNB = [
    value for sublist in df_filtered.SaccDirectionNoBias.values for value in sublist
]

# |%%--%%| <mX2Itr1fGW|YdwwYJc9jy>

dd = pd.DataFrame({"POS": POS, "MV": MV, "TD": TD, "SD": SD})
# Create a DataFrame
ddB = pd.DataFrame({"POS": POSB, "MV": MVB, "TD": TDB, "SD": SDB})
ddNB = pd.DataFrame({"POS": POSNB, "MV": MVNB, "TD": TDNB, "SD": SDNB})
dd.dropna(inplace=True)
ddB.dropna(inplace=True)
ddNB.dropna(inplace=True)

# |%%--%%| <YdwwYJc9jy|YZAj6SMUGa>

# Plot MV as a function of SD
sns.scatterplot(data=dd, x="SD", y="MV", hue="TD")

# |%%--%%| <YZAj6SMUGa|jtsOOXMPHL>

# Plot MV as a function of SD in Bias
sns.scatterplot(data=ddB, x="SD", y="MV", hue="TD")

# |%%--%%| <jtsOOXMPHL|ryWRzODAyc>

# Plot MV as a function of SD in NoBias
sns.scatterplot(data=ddNB, x="SD", y="MV", hue="TD")

# |%%--%%| <ryWRzODAyc|7IWx4Ct6uL>

# Statistical test on the Bias

stats.ttest_ind(ddB[ddB.SD == -1].MV, ddB[ddB.SD == 1].MV)

# |%%--%%| <7IWx4Ct6uL|YmX0AETsAW>

# Statistical test on the No Bias
stats.ttest_ind(ddNB[ddNB.SD == -1].MV, ddNB[ddNB.SD == 1].MV)

# |%%--%%| <YmX0AETsAW|bMmNzhETi8>

# Plot SD as a function of MV>0 and MV<0
plt.scatter(dd[dd.MV > 0].SD, dd[dd.MV > 0].MV, label="MV>0", alpha=0.5)
# plt.scatter(dd[dd.MV<0].SD,dd[dd.MV<0].MV,label="MV<0",alpha=0.5)
plt.legend()
plt.xlabel("Saccade Direction")
plt.ylabel("Mean Velocity")
plt.savefig("SaccadeDirectionEnfants")

# |%%--%%| <bMmNzhETi8|XyvHdcGdD3>

# Ration of the SD=1:
len(dd[(dd.SD == 1) & (dd.MV < 0)]) / len(dd[dd.MV < 0].SD)

# |%%--%%| <XyvHdcGdD3|RuF7KbQLaW>

dd[(dd.SD == -1) & (dd.MV < 0)]

# |%%--%%| <RuF7KbQLaW|Xs7nl5cRIE>

len(dd[dd.SD > 0].SD) / len(dd.SD)

# |%%--%%| <Xs7nl5cRIE|SgfUNmGyYo>

# Statistical test
stats.ttest_ind(dd[dd.SD == -1].MV, dd[dd.SD == 1].MV)

# |%%--%%| <SgfUNmGyYo|tbId1xKfrs>

plt.scatter(dd.MV, dd.SD, label="MV>0", alpha=0.5)
# plt.scatter(dd[dd.MV<0].SD,dd[dd.MV<0].MV,label="MV<0",alpha=0.5)
plt.legend()
plt.ylabel("Saccade Direction")
plt.xlabel("Mean Velocity")
plt.savefig("SaccadeDirectionEnfants")

# |%%--%%| <tbId1xKfrs|v3KL0BX6xe>

# Plot MV as a function of SD
sns.histplot(data=dd, x="MV", hue="SD")

# |%%--%%| <v3KL0BX6xe|by6P2KJQFd>

# Plot SD as a function of TD
sns.histplot(data=dd, x="SD", hue="TD")

# |%%--%%| <by6P2KJQFd|oUq0kXuGw1>

len(dd[(dd.MV > 0) & (dd.SD < 0)])

# |%%--%%| <oUq0kXuGw1|Vc7pjBJDQP>

len(dd[(dd.MV < 0) & (dd.SD > 0)])

# |%%--%%| <Vc7pjBJDQP|qPRmFJqw0x>

len(dd)

# |%%--%%| <qPRmFJqw0x|MaDrzgvsuH>

len(dd[(dd.MV > 0) & (dd.SD > 0)])

# |%%--%%| <MaDrzgvsuH|Cu13S87YDz>

len(dd[(dd.MV < 0) & (dd.SD < 0)])

# |%%--%%| <Cu13S87YDz|Si0dq9rFgb>


