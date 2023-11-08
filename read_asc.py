
import numpy as np 
import pandas as pd 
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from collections import defaultdict
from frites import set_mpl_style
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import io
import warnings
set_mpl_style()

# |%%--%%| <baHIxbUXs8|4UZ0ikAVB0>

def process_events(rows, blocks, colnames):
    
    # If no data, create empty dataframe w/ all cols and types
    if len(rows) == 0:
        rows = ["", ""]
        blocks = []
    # Parse data, dropping useless first column
    if len(rows) == 1:
        rows.append("")
    colnames = ['type'] + colnames # first col is event type, which we drop later
    coltypes = get_coltypes(colnames)
    df = pd.read_csv(io.StringIO('\n'.join(rows)), delimiter='\s+', header=None, names=colnames, na_values='.', index_col=False)
    df = df.iloc[:, 1:] # drop the first column
    # Move eye column to end & make factor, append block numbers to beginning of data frame
    if 'eye' in colnames:
        df = df.iloc[:, [1] + list(range(2, df.shape[1])) + [0]]
        df['eye'] = pd.Categorical(df['eye'], categories=["L", "R"], ordered=False)
    df.insert(loc=0, column='trial', value=blocks)
    return df




def process_saccades(saccades, blocks, info):
    sacc_df = process_events(saccades, blocks, get_sacc_header(info))
    # Set amplitudes for any saccades missing start/end coords to NAs because they're wonky
    ampl_cols = [col for col in sacc_df.columns if re.search(r'ampl\d*$', col)]
    partial = sacc_df['sxp'].isna() | sacc_df['exp'].isna()
    if any(partial):
        sacc_df.loc[partial, ampl_cols] = pd.NA
    return sacc_df



def process_fixations(fixations, blocks, info):
    return process_events(fixations, blocks, get_fix_header(info))


def process_blinks(blinks, blocks):
    return process_events(blinks, blocks, ['eye', 'stime', 'etime', 'dur'])


def process_messages(msgs, blocks):
    # Process messages from tracker
    msg_mat = [msg.split(' ', 1) for msg in msgs]
    msg_mat = [[msg[0][4:], msg[1]] for msg in msg_mat]
    msg_df = pd.DataFrame(msg_mat, columns=['time', 'text'])
    msg_df['time'] = pd.to_numeric(msg_df['time'])
    
    # Append trial numbers to beginning of data frame
    msg_df.insert(0, 'trial', blocks)
    
    return msg_df


def process_input(input_data, blocks):
    return process_events(input_data, blocks, ['time', 'value'])


def process_buttons(button, blocks):
    return process_events(button, blocks, ['time', 'button', 'state'])


# |%%--%%| <4UZ0ikAVB0|7hlM0JcuzO>

import re

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


# |%%--%%| <7hlM0JcuzO|hV5CG5NSzM>


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
        "BLRR": "Long-Range Mount / Binocular / Head Stabilized"
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
    
    


# |%%--%%| <hV5CG5NSzM|WICWxCJLHd>

def get_model(header):
    version_str = from_header(header, "VERSION")
    version_str2 = [x for x in header if re.search("\\*\\* EYELINK II", x)]
    if version_str is None:
        model = "Unknown"
        ver_num = "Unknown"
    elif version_str != 'EYELINK II 1':
        model = "EyeLink I"
        ver_num = re.search(r'(\d+.\d+)', version_str).group(1)
    else:
        ver_num = re.search(r'v(\d+.\d+)', version_str2[0]).group(1)
        model = ('EyeLink II' if float(ver_num) < 2.4 else
                 'EyeLink 1000' if float(ver_num) < 5 else
                 'EyeLink 1000 Plus' if float(ver_num) < 6 else
                 'EyeLink Portable Duo')
    return [model, ver_num]

# |%%--%%| <WICWxCJLHd|zsKbPepXLt>

def get_coltypes(colnames, float_time=True):
    chr_cols=["type", "eye", "cr.info", "remote.info"]
    int_cols=["button", "state", "value"]
    time_cols=["time", "stime", "etime", "dur"]
    if not float_time:
        int_cols += time_cols
    
    coltypes = ['str' if col in chr_cols else 'int64' if col in int_cols else 'float64' for col in colnames]
    
    return coltypes


def get_htarg_regex(binocular):
    htarg_errs = "MANCFTBLRTBLRTBLR" if binocular else "MANCFTBLRTBLR"
    htarg_errs = list(htarg_errs)
    htarg_regex = "(" + "|".join(htarg_errs + ["\\."]) + ")"
    
    return htarg_regex


def is_float(string):
    return bool(re.search("\\.", string))

# |%%--%%| <zsKbPepXLt|9Gj3pdtgzt>

def get_info(nonsample, firstcol):
    header = [f for f in nonsample if f.startswith ("**")]
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

    #Get pupil size data type (area or diameter)
    pupil_config = [line for i,line in enumerate(nonsample) if firstcol[i] == "PUPIL"]
    if len(pupil_config) > 0:
        info["pupil.dtype"] = pupil_config[-1].split()[1]

    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [line for i,line in enumerate(nonsample) if firstcol[i] == "EVENTS"]
    samples_config = [line for i,line in enumerate(nonsample) if firstcol[i]== "SAMPLES"]
    
    # Find the samples and events config lines in the non-sample input, get data types
    events_config = [line for i,line in enumerate(nonsample) if firstcol[i] == "EVENTS"]
    samples_config = [line for i,line in enumerate(nonsample) if firstcol[i] == "SAMPLES"]
    if len(events_config) > 0:
        info["event.dtype"] = events_config[-1].split()[1]
    if len(samples_config) > 0:
        info["sample.dtype"] = samples_config[-1].split()[1]
    
   
        
    
    # Get last config line in file (preferring sample config) and extract remaining info
    config = events_config + samples_config[-1:]
    config = config[-1] if len(config) > 0 else ""
    if config:
        info["sample.rate"] = float(re.findall(r"RATE\s+([0-9]+\.[0-9]+)", config)[0]) if "RATE" in config else None
        info["tracking"] = "\tTRACKING" in config
        info["cr"] = "\tCR" in config
        info["filter.level"] = int(re.findall(r"FILTER\s+([0-9]+)", config)[0]) if "FILTER" in config else None
        info["velocity"] = "\tVEL" in config
        info["resolution"] = "\tRES" in config
        info["htarg"] = "\tHTARG" in config
        info["input"] = "\tINPUT" in config
        info["buttons"] = "\tBUTTONS" in config
        info["left"] = "\tLEFT" in config
        info["right"] = "\tRIGHT" in config
        info["mono"] = not(info["right"] & info["left"])



    return info


# |%%--%%| <9Gj3pdtgzt|CjCEn2oZkg>

def process_raw(raw, blocks, info):
    if len(raw) == 0:
        # If no sample data in file, create empty raw DataFrame w/ all applicable columns
        raw = ["", ""]
        blocks = pd.Series([], dtype=int)
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time=False)
    else:
        # Determine if timestamps stored as floats (edf2asc option -ftime, useful for 2000 Hz)
        float_time = is_float(re.split(r'\s+', raw[0])[0])
        # Generate column names and types based in info in header
        colnames = get_raw_header(info)
        coltypes = get_coltypes(colnames, float_time)
        # Discard any rows with too many or too few columns (usually rows where eye is missing)
        row_length = [len(re.split(r'\t', r)) for r in raw]
        med_length = np.median(row_length)
        raw = [r for r, l in zip(raw, row_length) if l == med_length]
        blocks = blocks[row_length == med_length]
        # Verify that generated columns match up with actual maximum row length
        length_diff = med_length - len(colnames)
        if length_diff > 0:
            warnings.warn("Unknown columns in raw data. Assuming first one is time, please check the others")
            colnames = ["time"] + [f"X{i+1}" for i in range(med_length-1)]
            coltypes = "i" + "?"*(med_length-1)
    # Process raw sample data using pandas
    if len(raw) == 1:
        raw.append("")
   
    raw_df = pd.read_csv(io.StringIO("".join(raw)), sep='\t', header=None, names=colnames, na_values=np.nan, low_memory=False)

    if info["tracking"] and not info["cr"]:
        raw_df = raw_df.drop(columns=["cr.info"]) # Drop CR column when not actually used
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


# |%%--%%| <CjCEn2oZkg|NHXnZtoErv>

def read_asc(fname, samples=True, events=True, parse_all=False):
    with open(fname, 'r') as f:
        inp = f.readlines()

    # Convert to ASCII
    inp = [line.encode('ascii', 'ignore').decode() for line in inp]
    
    # Get strings prior to first tab for each line for faster string matching
    inp_first = [re.split(r'\s', s)[0] for s in inp]
    
    #Get the Trial info for each trial: 
    bias = [s.split()[4]for s in inp if len(s.split()) > 4 and s.split()[2] == 'Trialinfo:']
    direct = [s.split()[5]for s in inp if len(s.split()) > 4 and s.split()[2] == 'Trialinfo:']
    # Check if any actual data recorded in file
    starts = [i for i,x in enumerate(inp_first) if x == "START"]
    if not starts:
        raise ValueError("No samples or events found in .asc file.")

    # Read metadata from file before processing
    is_raw = [bool(re.match('^[0-9]', line)) for line in inp_first]
    
    info = get_info([line for line, raw in zip(inp, is_raw) if not raw],
                    [first for first, raw in zip(inp_first, is_raw) if not raw])

    # Do some extra processing/sanitizing if there's HTARG info in the file
    if info['htarg']:
        inp, info = handle_htarg(inp, info, is_raw)

    # Find blocks and mark lines between block ENDs and next block STARTs
    dividers = starts + [len(inp)]
    block = np.cumsum([x == "START" for x in inp_first])
    block = block.astype(float)

    for i in range(1, len(dividers)):
        start = dividers[i-1]
        end = dividers[i]
        endline = [j for j, x in enumerate(inp_first[start:end]) if x == "END"]
        if endline and endline[-1] < end-start:
             block[endline[0]+start:end] += 0.5

    # Unless parsing all input, drop any lines not within a block
    block[:dividers[0]+1] += 0.5
    if not parse_all:
        in_block = np.floor(block) == block
        inp = [line for line, block_match in zip(inp, in_block) if block_match]
        inp_first = [first for first, block_match in zip(inp_first, in_block) if block_match]
        is_raw = [raw for raw, block_match in zip(is_raw, in_block) if block_match]
        block = block[in_block]
    
    block=np.array(block)

    # Initialize dictionary of data output and process different data types
    out = {}
    if samples:
        out['raw'] = process_raw([line for line, raw in zip(inp, is_raw) if raw],
                                 block[is_raw], info)
    if events:
        is_sacc = np.array(inp_first) == "ESACC"
        out['sacc'] = process_saccades(np.array(inp)[is_sacc], np.array(block)[is_sacc], info)

        is_fix = np.array(inp_first) == "EFIX"
        out['fix'] = process_fixations(np.array(inp)[is_fix], np.array(block)[is_fix], info)

        is_blink = np.array(inp_first) == "EBLINK"
        out['blinks'] = process_blinks(np.array(inp)[is_blink], np.array(block)[is_blink])

        is_msg = np.array(inp_first) == "MSG"
        out['msg'] = process_messages(np.array(inp)[is_msg], np.array(block)[is_msg])

        is_input = np.array(inp_first) == "INPUT"
        out['input'] = process_input(np.array(inp)[is_input], np.array(block)[is_input])

        is_button = np.array(inp_first) == "BUTTON"
        out['button'] = process_buttons(np.array(inp)[is_button], np.array(block)[is_button])

    
    info['tracking'] = None  # needed for parsing, but otherwise redundant with CR
    
    out['info'] = info

    return out,np.array(bias,dtype='int'), np.array(direct,dtype='int')%2


# |%%--%%| <NHXnZtoErv|x1TMIdgBPi>
r"""°°°
# Processing data from 1 subject
°°°"""
# |%%--%%| <x1TMIdgBPi|HIBotD1Nw6>

f='/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/Ados/ABBO/ABBOa90.asc'

# |%%--%%| <HIBotD1Nw6|h9NftQ5MEl>

data,bias,direct=read_asc(f)

# |%%--%%| <h9NftQ5MEl|zvWJDlBR9A>

data.keys()

# |%%--%%| <zvWJDlBR9A|i5EzmEOZ4W>

data["info"]

# |%%--%%| <i5EzmEOZ4W|6j0WCNOY2u>

np.where(bias==1)[0][0]


# |%%--%%| <6j0WCNOY2u|LD6drbnHTy>

data["sacc"]

# |%%--%%| <LD6drbnHTy|0LnaUXqlQL>

df=data["raw"]

# |%%--%%| <0LnaUXqlQL|Oz3mxOpQ35>

MSG=data["msg"]
Zero=MSG.loc[MSG.text=='TargetOn\n',["trial","time"]]

# |%%--%%| <Oz3mxOpQ35|pW4E8xaR34>

Zero

# |%%--%%| <pW4E8xaR34|5Id3EXReNQ>

df[(df.time>=80) & (df.time<=120) & (df.trial==47)]

# |%%--%%| <5Id3EXReNQ|yGUwcBU83z>

df[(df.trial==49)]

# |%%--%%| <yGUwcBU83z|CPbYiSHs0g>

#df.loc[df.trial==150,'time'].values-Zero.loc[Zero.trial==150].time.values

# |%%--%%| <CPbYiSHs0g|cmroT0zB1J>

#Resetting the time
for t in Zero.trial:
    df.loc[df.trial==t,'time']= df.loc[df.trial==t,'time'].values-Zero.loc[Zero.trial==t,'time'].values

# |%%--%%| <cmroT0zB1J|pTT7swy7cF>

for t in np.unique(df.trial):
    l=len(df[(df.time>=80) & (df.time<=120) & (df.trial==t) ])
    if l<41:
        print(t)
    

# |%%--%%| <pTT7swy7cF|GngGQfd2Wq>

#Checking if the experiment was binorcular or monocular
mono=data["info"]["mono"]

# |%%--%%| <GngGQfd2Wq|tIL4i29OjC>


df['trial'] = pd.to_numeric(df['trial'], errors='coerce')
df['time'] = pd.to_numeric(df['time'], errors='coerce')
if not mono:
    df['xpl'] = pd.to_numeric(df['xpl'], errors='coerce')
    df['ypl'] = pd.to_numeric(df['ypl'], errors='coerce')
    df['psl'] = pd.to_numeric(df['psl'], errors='coerce')
    df['xpr'] = pd.to_numeric(df['xpr'], errors='coerce')
    df['ypr'] = pd.to_numeric(df['ypr'], errors='coerce')
    df['psr'] = pd.to_numeric(df['psr'], errors='coerce')
else:
    df['xp'] = pd.to_numeric(df['xp'], errors='coerce')
    df['ypl'] = pd.to_numeric(df['yp'], errors='coerce')
    df['ps'] = pd.to_numeric(df['ps'], errors='coerce')
    
df['input'] = pd.to_numeric(df['input'], errors='coerce')

# |%%--%%| <tIL4i29OjC|Muip17jEQm>

data["fix"]

# |%%--%%| <Muip17jEQm|9dHtVpFIbL>

data["msg"]

# |%%--%%| <9dHtVpFIbL|E6W65ckBHn>

data["blinks"]

# |%%--%%| <E6W65ckBHn|WhrS4lj5uf>

t=df[df.trial==20].time.values
if mono:
    pos=df[df.trial==20].xp
else:
    pos=df[df.trial==20].xpr
    
plt.scatter(t,pos)
plt.xlabel("Time (ms)")
plt.ylabel ("Eye position along x-axis (px)")
#_ = plt.yticks(np.arange(0,1680,100))

# |%%--%%| <WhrS4lj5uf|4tk5oseEeC>

MSG=data["msg"]
MSG.head()

# |%%--%%| <4tk5oseEeC|ymKtnF6lE9>

t0=MSG.loc[MSG.text=='StimulusOff\n','time']
Zero=MSG.loc[MSG.text=='TargetOn\n',["trial","time"]]

# |%%--%%| <ymKtnF6lE9|u0bwosIgS9>
r"""°°°
### Time Rescale:
°°°"""
# |%%--%%| <u0bwosIgS9|SOKnIVmq1S>

for i in range (len(Zero)):
    df.loc[df['trial'] == i+1, 'time'] = df.loc[df['trial'] == i+1, 'time'] - Zero.time.values[i]

# |%%--%%| <SOKnIVmq1S|m4ruQSG8f1>

df.time

# |%%--%%| <m4ruQSG8f1|UbA13AGBc6>

t0=t0.values-Zero.time.values

# |%%--%%| <UbA13AGBc6|1JaBGRAtaC>

for i in range(99,150):
    if not mono:
        plt.subplot(1,2,1)
        plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])
        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]))
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()
    else:
        plt.subplot(1,2,1)
        plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])
        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]))
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()

# |%%--%%| <1JaBGRAtaC|5fg88C7TfE>

bias

# |%%--%%| <5fg88C7TfE|71HGB8Fo3w>

first_bias=np.where(bias==1)[0][0]+1

# |%%--%%| <71HGB8Fo3w|CwC8vqFGs2>

direct

# |%%--%%| <CwC8vqFGs2|SxUYLMERx9>


if not mono:
    dyn1=df.xpr[(df.time>=250) & (df.time<=300)& (df.trial<first_bias)]
    dyn2=df.xpr[(df.time>=250) & (df.time<=300)& (df.trial>=first_bias)]
else:
    dyn1=df.xp[(df.time>=250) & (df.time<=300)& (df.trial<first_bias)]
    dyn2=df.xp[(df.time>=250) & (df.time<=300)& (df.trial>=first_bias)]

# |%%--%%| <SxUYLMERx9|lPht2piBP2>


# compute the histogram
H1 = []
H2 = []
for t in range(-500, 750, 50):
    if not mono:
        dyn1 = df.xpr[(df.time >= t) & (df.time <= t + 50) & (df.trial < 60)]
        dyn2 = df.xpr[(df.time >= t) & (df.time <= t + 50) & (df.trial >= 60)]
        h1 = plt.hist(dyn1, alpha=.5, label='non bias', bins=range(400, 1200, 10))
        h2 = plt.hist(dyn2, alpha=.5, label='bias', bins=range(400, 1200, 10))
        H1.append(h1[0])
        H2.append(h2[0])
    else:
        dyn1 = df.xp[(df.time >= t) & (df.time <= t + 50) & (df.trial < 60)]
        dyn2 = df.xp[(df.time >= t) & (df.time <= t + 50) & (df.trial >= 60)]
        h1 = plt.hist(dyn1, alpha=.5, label='non bias', bins=range(400, 1200, 10))
        h2 = plt.hist(dyn2, alpha=.5, label='bias', bins=range(400, 1200, 10))
        H1.append(h1[0])
        H2.append(h2[0])
        


# |%%--%%| <lPht2piBP2|QIBZHF6TEb>

H1 = np.array(H1)
H2 = np.array(H2)

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

# Plot the first heatmap
im1 = axs[0].imshow(H1.T, cmap='coolwarm', origin='lower')
axs[0].set_xticks(np.arange(0, H1.shape[0], 5))
axs[0].set_xticklabels(np.arange(-500, 750, 250))
axs[0].set_yticks(np.arange(0, H1.shape[1], 10))
axs[0].set_yticklabels(np.arange(400, 1200, 100))
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('X position')
axs[0].set_title('Non Bias')

# Plot the second heatmap
im2 = axs[1].imshow(H2.T, cmap='coolwarm', origin='lower')
axs[1].set_xticks(np.arange(0, H2.shape[0], 5))
axs[1].set_xticklabels(np.arange(-500, 750, 250))
axs[1].set_yticks(np.arange(0, H2.shape[1], 10))
axs[1].set_yticklabels(np.arange(400, 1200, 100))
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('X position')
axs[1].set_title('Bias')

# Add colorbars to the heatmaps
cbar1 = fig.colorbar(im1, ax=axs[0])
cbar1.set_label('Counts')
cbar2 = fig.colorbar(im2, ax=axs[1])
cbar2.set_label('Counts')
plt.tight_layout()
plt.show()
plt.savefig('hm100')

# |%%--%%| <QIBZHF6TEb|YnpfLK6qWG>

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Initialize the figure and axes
fig, ax = plt.subplots()

# Set the x-axis limits and bin size
bin_size = 20

# Create the initial histograms
if not mono:
    data1 = df.xpr[(df.time>=-500) & (df.time<=300) & (df.trial<60)]
    data2 = df.xpr[(df.time>=-500) & (df.time<=300) & (df.trial>=60)]

else:
    data1 = df.xp[(df.time>=-500) & (df.time<=300) & (df.trial<60)]
    data2 = df.xp[(df.time>=-500) & (df.time<=300) & (df.trial>=60)]
#Fixed values for xax and xmin
x_max = int(np.max([data1.max(), data2.max()])) + 10
x_min = int(np.min([data1.min(), data2.min()])) -10

hist1, bins1 = np.histogram(data1, bins=range(x_min, x_max + bin_size, bin_size))
rects1 = ax.bar(bins1[:-1], hist1, width=bin_size, align='edge', alpha=0.5,label="non bias")

hist2, bins2 = np.histogram(data2, bins=range(x_min, x_max + bin_size, bin_size))
rects2 = ax.bar(bins2[:-1], hist2, width=bin_size, align='edge', alpha=0.5,label="bias")
ax.set_ylim([0, 1000])

plt.legend()

# Define the update function
time_start = -500
time_end = 800
interval = 20

def update(frame):
    global time_start
    if not mono:
        data1 = df.xpr[(df.time >= time_start) & (df.time <= time_start + interval) & (df.trial < 60)]
        data2 = df.xpr[(df.time >= time_start) & (df.time <= time_start + interval) & (df.trial >= 60)]
    else:
        data1 = df.xp[(df.time >= time_start) & (df.time <= time_start + interval) & (df.trial < 60)]
        data2 = df.xp[(df.time >= time_start) & (df.time <= time_start + interval) & (df.trial >= 60)]
    
    time_start += interval
    
    hist1, bins1 = np.histogram(data1, bins=range(x_min, x_max + bin_size, bin_size))
    for rect1, h1 in zip(rects1, hist1):
        rect1.set_height(h1)

    hist2, bins2 = np.histogram(data2, bins=range(x_min, x_max + bin_size, bin_size))
    for rect2, h2 in zip(rects2, hist2):
        rect2.set_height(h2)
    
    return rects1, rects2

# Create the animation
frames = int((time_end - time_start) / interval)  # calculate number of frames
ani = FuncAnimation(fig, update, frames=frames, interval=interval, repeat=True)

# Display the animation
HTML(ani.to_jshtml())



# |%%--%%| <YnpfLK6qWG|28Fg2VlFC5>

Sacc=data["sacc"]

# |%%--%%| <28Fg2VlFC5|JMByIStY4M>
r"""°°°
## Rescaling time of the saccades
°°°"""
# |%%--%%| <JMByIStY4M|GNGU6ksWOQ>

for t in Zero.trial:
    Sacc.loc[Sacc.trial==t,["stime","etime"]]=Sacc.loc[Sacc.trial==t,["stime","etime"]].values-Zero.loc[Zero.trial==t,'time'].values
  

# |%%--%%| <GNGU6ksWOQ|7CidwMSCjU>

Sacc

# |%%--%%| <7CidwMSCjU|iyfFtwGqy5>


for t in Sacc.trial.unique():
    
    start=Sacc.loc[(Sacc.trial==t) & (Sacc.eye=="R"),"stime"]
    end=Sacc.loc[(Sacc.trial==t) & (Sacc.eye=="R"),"etime"]

    for i in range(len(start)):
        if not mono:
            df.loc[(df.trial==t) & (df.time>=start.iloc[i]-20) & (df.time<=end.iloc[i]+20),'xpr']=np.nan
        else:
             df.loc[(df.trial==t) & (df.time>=start.iloc[i]-20) & (df.time<=end.iloc[i]+20),'xp']=np.nan
            

# |%%--%%| <iyfFtwGqy5|fnRpZNvfEW>

for i in range(150):
    if not bias[i]:
        plt.subplot(1,2,1)
        if not mono:
            plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])
        else:
            plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])

        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]))
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]))
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="t0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()
    else:
        plt.subplot(1,2,1)
        if not mono:
            plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],color="pink")
        else:
            plt.scatter(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],color="pink")

        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),color="pink")
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],np.gradient(df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),color="pink")

            
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()
        

# |%%--%%| <fnRpZNvfEW|Xs87xKFzAe>

from scipy.ndimage import gaussian_filter1d

# |%%--%%| <Xs87xKFzAe|kxZDKhm0b7>

for i in range(150):
    if not bias[i]:
        plt.subplot(1,2,1)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)])

        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],gaussian_filter1d(np.gradient(df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),sigma=4))
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],gaussian_filter1d(np.gradient(df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),sigma=4))
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()
    else:
        plt.subplot(1,2,1)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],color="black")
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300) & (df.trial==i+1)],df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],color="black")

        plt.vlines(df.time[df.time==100],800,950,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],800,950,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],800,950,colors='black',label="t0",lw=2)
        plt.legend()
        plt.subplot(1,2,2)
        if not mono:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],gaussian_filter1d(np.gradient(df.xpr[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),sigma=10),color="black")
        else:
            plt.plot(df.time[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)],gaussian_filter1d(np.gradient(df.xp[(df.time>=-500) & (df.time<=300)& (df.trial==i+1)]),sigma=10),color="black")

            
        plt.vlines(df.time[df.time==100],-1,1,colors='blue',lw=2,label="tapp")
        plt.vlines(df.time[df.time==0],-1,1,colors='green',label="0",lw=2)
        plt.vlines(df.time[df.time==t0[i]],-1,1,colors='black',label="t0",lw=2)
        plt.legend()
        plt.show()

# |%%--%%| <kxZDKhm0b7|qCKoF9YMDS>



# |%%--%%| <qCKoF9YMDS|I9LreMPyXz>


