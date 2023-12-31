
import numpy as np 
from scipy.stats import linregress,pearsonr,normaltest
import pandas as pd 
import re
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from collections import defaultdict
from frites import set_mpl_style
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns
import warnings
import io
from scipy import stats
import xarray as xr
set_mpl_style()
import json

# |%%--%%| <ioPVUjc6yG|TdSqQ3LvtX>

def process_events(rows, blocks, colnames):
    
    # If no data, create empty dataframe w/ all cols and types
    if len(rows) == 0:
        rows = ["", ""]
        blocks = []
    # Parse data, dropping useless first column
    if len(rows) == 1:
        list(rows).append("")
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

# |%%--%%| <TdSqQ3LvtX|xl6yfE3rIx>

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

# |%%--%%| <xl6yfE3rIx|UeoPhj7a0U>

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
    
    


# |%%--%%| <UeoPhj7a0U|3aE1nxGpoT>

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

# |%%--%%| <3aE1nxGpoT|h1DU8HELrC>

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

# |%%--%%| <h1DU8HELrC|ec8SctsLuY>

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



# |%%--%%| <ec8SctsLuY|z7bhfOf6IL>

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
        #if length_diff > 0:
        #    warnings.warn("Unknown columns in raw data. Assuming first one is time, please check the others")
        #    colnames = ["time"] + [f"X{i+1}" for i in range(med_length-1)]
        #    coltypes = "i" + "?"*(med_length-1)
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


# |%%--%%| <z7bhfOf6IL|wdLHSLlVfl>

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


# |%%--%%| <wdLHSLlVfl|LP6XWQnlLq>

path='/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/'
from os import listdir
categories = np.sort([f for f in listdir(path)])
namesCat=[]
for cat in categories:
    namesCat.append(np.sort([f for f in listdir(path+cat)]))


# |%%--%%| <LP6XWQnlLq|fAp6YFkaPE>

allFiles=[]

for nameCat,cat in zip(namesCat,categories):
    filesCat=[]
    for name in nameCat:
        files=np.sort([f for f in listdir(path+cat+'/'+name)])
        filesCat.append(files)
    allFiles.append(filesCat)


# |%%--%%| <fAp6YFkaPE|jHL5c7pfl5>

allPaths=[]
for cat,names,conditions in zip(categories,namesCat,allFiles):
    catPaths=[]
    for name,namecond in zip(names,conditions):
        for condition in namecond:
            catPaths.append(path+cat+'/'+name+'/'+condition)
    allPaths.append(catPaths)

# |%%--%%| <jHL5c7pfl5|MfIMNww0Xc>



# |%%--%%| <MfIMNww0Xc|PD2cYmQzbh>
r"""°°°
## Going through all the Adults

°°°"""
# |%%--%%| <PD2cYmQzbh|OP134wEuOB>

meanPos=[]
stdPos=[]


meanVelo=[]
stdVelo=[]



switch=[]

Probas=[]
Directions=[]

TS=[]
SaccD=[]

for f in allPaths[1]:
    data,bias,direct=read_asc(f)
    
    #Getting the probability from the name of the file
    proba=int(f[-6:-4])/100#probability
    if proba==0.0:
        proba=1
    
    df=data["raw"]
    
    #Checking if the experiment was binorcular or monocular
    mono=data["info"]["mono"]
    #Putting all the data in the right format
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
    
    #Messages from eyelink:
    MSG=data["msg"]
    t0=MSG.loc[MSG.text=='StimulusOff\n','time']
    Zero=MSG.loc[MSG.text=='TargetOn\n',["trial","time"]]
    
    
    #resetting the time
    for i in range (len(Zero)):
        df.loc[df['trial'] == i+1, 'time'] = df.loc[df['trial'] == i+1, 'time'] - Zero.time.values[i]
    
    #Getting the saccades:
    Sacc=data["sacc"]
    
    #Resetting the saccades
    for t in Zero.trial:
        Sacc.loc[Sacc.trial==t,["stime","etime"]]=Sacc.loc[Sacc.trial==t,["stime","etime"]].values-Zero.loc[Zero.trial==t,'time'].values
        
    
    
         
    ###Getting the trials where the saccades happends inside the time window. 0 and 80ms.
    trialSacc=Sacc[(Sacc.stime >= 0) & (Sacc.etime <80) & (Sacc.eye=='R')]["trial"].values
    
    saccDir=np.sign((Sacc[(Sacc.stime >= 0) & (Sacc.etime < 80) &  (Sacc.eye == 'R')].exp - Sacc[(Sacc.stime >=0) & (Sacc.etime < 80)  &  (Sacc.eye == 'R')].sxp).values)
        
  
    for t in Sacc.trial.unique():
    
        start=Sacc.loc[(Sacc.trial==t) & (Sacc.eye=="R"),"stime"]
        end=Sacc.loc[(Sacc.trial==t) & (Sacc.eye=="R"),"etime"]
        
   

        for i in range(len(start)):

            if not mono:




                df.loc[(df.trial==t) & (df.time>=start.iloc[i]-20) & (df.time<=end.iloc[i]+20),'xpr']=np.nan



            else:



                df.loc[(df.trial==t) & (df.time>=start.iloc[i]-20) & (df.time<=end.iloc[i]+20),'xp']=np.nan


    
    
    
    
    
    #first porbability switch
    first_bias=np.where(bias==1)[0][0]
    switch.append(first_bias)
    
    if not mono:
        # Select the desired values
        selected_values = df.xpr[(df.time >= 80) & (df.time <=120)]
        
        #print(len(selected_values))
       
        #Rescale the position: 
        pos_before= df.xpr[(df.time >=-40) & (df.time <=0)]

        
        # Reshape into a 2D matrix
        time_dim = 41
        
        trial_dim = len(selected_values) // time_dim
        
        #Re-basing the position
        pos= np.array(selected_values[:time_dim * trial_dim]).reshape(trial_dim, time_dim)
        #variance of pos on bias and non bias trials
        stdPos.append(np.std(pos,axis=1)/30)
        pos_before_reshaped=np.array(pos_before[:time_dim * trial_dim]).reshape(trial_dim, time_dim)
        pos_before_mean=np.nanmean(pos_before_reshaped,axis=1)
        
        
        velo =np.gradient(pos,axis=1)*1000/30
        velo[(velo>30) | (velo<-30)]=np.nan

        
        for i,pp in enumerate(pos_before_mean):
            if pp==np.nan:
                pos[i]=np.nan
            else:
                pos[i]=(pos[i]-pp)/30

                
        #pos=(pos-pos_before_mean.reshape(-1, 1))/30
        pos[(pos>3) | (pos<-3)]=np.nan
       
        
        
        
        
    else:
        
        # Select the desired values
        selected_values = df.xp[(df.time >= 80) & (df.time <= 120)]
        #print(len(selected_values))
        
        #Rescale the position: 
        pos_before= df.xp[(df.time >=-40) & (df.time <=0)]

        # Reshape into a 2D matrix
        time_dim = 41
        
        trial_dim = len(selected_values) // time_dim
        
        
        
        pos = np.array(selected_values[:time_dim * trial_dim]).reshape(trial_dim, time_dim)
        stdPos.append(np.std(pos,axis=1)/30)
        pos_before_reshaped=np.array(pos_before[:time_dim * trial_dim]).reshape(trial_dim, time_dim)
        pos_before_mean=np.nanmean(pos_before_reshaped,axis=1)
        
        
            
        velo =np.gradient(pos,axis=1)*1000/30
        velo[(velo>30) | (velo<-30)]=np.nan
        
        for i,pp in enumerate(pos_before_mean):
            if pp==np.nan:
                pos[i]=np.nan
            else:
                pos[i]=(pos[i]-pp)/30
            
        
        #pos=(pos-pos_before_mean.reshape(-1, 1))/30
        pos[(pos>3) | (pos<-3)]=np.nan
        
        
 
    #mean pos on bias and non bias trials for time window 80,120
    meanPos.append(np.mean(pos,axis=1))
   
    
    
    #mean of velocity on bias and non bias trials
    meanVelo.append(np.mean(velo,axis=1))
    
    #var of velocity on bias and non bias trials
    stdVelo.append(np.std(velo,axis=1))
    
    #subjects.append(name)
    
    Probas.append(proba)
    
    
    #Adding target directions 
    
    Directions.append(direct)
    
    #Trials where there is a saccade in the time window [0, 80ms]
    
    TS.append(trialSacc)
    
    #Direction of the saccad
    
    SaccD.append(saccDir)
    
    


# |%%--%%| <OP134wEuOB|5NBONekWZ8>

Sacc

# |%%--%%| <5NBONekWZ8|86G4i0uvgJ>

Sacc[(Sacc.stime > 0) & (Sacc.etime < 80)  &  (Sacc.eye == 'R')]["dur"].values

# |%%--%%| <86G4i0uvgJ|7xOuZORdMh>

start.values 

# |%%--%%| <7xOuZORdMh|rs5JXbFBVz>

end.values

# |%%--%%| <rs5JXbFBVz|Cgd2m4sC3n>


df= pd.DataFrame({"meanPos": meanPos,"stdPos": stdPos, "meanVelo": meanVelo,"stdVelo": stdVelo,"proba": Probas,"switch":switch,"tgt_direction":Directions,"SaccDirection":SaccD,"SaccTrial":TS})
subjects=[]
for name,file in zip (namesCat[1],allFiles[1]):
    for i in range(len(file)):
        subjects.append(name)
df["name"]=subjects

# |%%--%%| <Cgd2m4sC3n|90eSyJey4c>

# Calculate the mean for 'meanPos' starting from the corresponding 'switch'
df['meanPosBias'] = df.apply(lambda row: pd.Series(row['meanPos'][row['switch']:]).mean(), axis=1)
df['meanPosNoBias'] = df.apply(lambda row: pd.Series(row['meanPos'][:row['switch']]).mean(), axis=1)
# Calculate the std for 'meanPos' starting from the corresponding 'switch'
df['stdPosBias'] = df.apply(lambda row: pd.Series(row['meanPos'][row['switch']:]).std(), axis=1)
df['stdPosNoBias'] = df.apply(lambda row: pd.Series(row['meanPos'][:row['switch']]).std(), axis=1)

# |%%--%%| <90eSyJey4c|UQjyNNGtmw>

# Calculate the mean for 'meanPos' starting from theswitch
df['meanVeloBias'] = df.apply(lambda row:pd.Series(row['meanVelo'][row['switch']:]).mean(), axis=1)
df['meanVeloNoBias'] = df.apply(lambda row: pd.Series(row['meanVelo'][:row['switch']]).mean(), axis=1)

# Calculate the mean for 'stdPos' for the bias portion and nonbias portion
df['stdVeloBias'] = df.apply(lambda row: pd.Series(row['meanVelo'][row['switch']:]).std(), axis=1)
df['stdVeloNoBias'] = df.apply(lambda row: pd.Series(row['meanVelo'][:row['switch']]).std(), axis=1)

# |%%--%%| <UQjyNNGtmw|AyrmHcYUUx>

# Identify columns with lists or NumPy arrays
list_columns = [col for col in df.columns if isinstance(df[col][0], (list, np.ndarray))]

# Serialize columns with lists/arrays to JSON strings
for col in list_columns:
    if isinstance(df[col][0], np.ndarray):
        df[col] = df[col].apply(lambda x: json.dumps(x.tolist()))
    else:
        df[col] = df[col].apply(json.dumps)



# |%%--%%| <AyrmHcYUUx|NEWcFbt4vA>

df.head()

# |%%--%%| <NEWcFbt4vA|smenhClo9r>

df.meanPos[0]

# |%%--%%| <smenhClo9r|f8OxARw1T2>

# Save the DataFrame to a CSV file
df.to_csv('Adults.csv', index=False)

# |%%--%%| <f8OxARw1T2|feg5OOH4DR>
r"""°°°
## No need to run the Function again 
°°°"""
# |%%--%%| <feg5OOH4DR|c2eQmiivfA>

# Read the CSV file
list_columns=['meanPos',
 'stdPos',
 'meanVelo',
 'stdVelo',
 'tgt_direction',
 'SaccDirection',
 'SaccTrial']
df = pd.read_csv('Adults.csv')

# |%%--%%| <c2eQmiivfA|H5AySefWF7>

df.head()

# |%%--%%| <H5AySefWF7|QZgz6Z6Ifr>

plt.plot(df.meanPos[0])

# |%%--%%| <QZgz6Z6Ifr|aGUILhOcxy>

# Deserialize columns with lists/arrays back to their original data type
for col in list_columns:
    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)


# |%%--%%| <aGUILhOcxy|um2TkIX0Ma>

df.meanPos[0]

# |%%--%%| <um2TkIX0Ma|vyhhErEHyz>



# |%%--%%| <vyhhErEHyz|re1c796y3z>

df[df['meanPos'].apply(lambda x: len(x)) == 150].name.values

# |%%--%%| <re1c796y3z|YTaAd5gbn6>

nonValidVeloName=df[df.meanVeloBias.isna()].name.values

# |%%--%%| <YTaAd5gbn6|hPCHWxVbyp>

nonValidVeloName

# |%%--%%| <hPCHWxVbyp|IlN7DeFkAh>

nonValidPosName=df[df.meanPosBias.isna()].name.values

# |%%--%%| <IlN7DeFkAh|DayCAw7Y8j>

nonValidPosName

# |%%--%%| <DayCAw7Y8j|qj0NYLJzjk>

p_values=[]
pbs=[]
for i,s in enumerate(df.switch.values):
    vv=np.array(df.iloc[i].meanVelo[s:])
    if(len(vv)>0):
        vv=vv[~np.isnan(vv)]
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vv)
        #print("Bias test statistic:", jarque_bera_stat)
        print("Bias p-value:", jarque_bera_p_value)
        p_values.append(jarque_bera_p_value)
        pbs.append(df.iloc[i].proba)
        plt.hist(vv,label="Velo Bias",alpha=.5)
    
    #Non Bias block
    vvv=np.array(df.iloc[i].meanVelo[:s])
    if len(vvv)>0:
        vvv=vvv[~np.isnan(vvv)]
        plt.hist(vvv,label="Velo no Bias",alpha=.5)
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vvv)
        
    #print("No Bias test statistic:", jarque_bera_stat)
        print("No Bias p-value:", jarque_bera_p_value)
    plt.title(df.iloc[i]["name"]+', '+str(df.iloc[i].proba))
    plt.legend()
    plt.show()

# |%%--%%| <qj0NYLJzjk|PXnP9wvMcf>

PVS=pd.DataFrame({'pvalue':p_values,"proba":pbs})
sns.boxplot(data=PVS,x="proba",y="pvalue")

# |%%--%%| <PXnP9wvMcf|7xlGswNfjf>

len(PVS[(PVS.pvalue<0.01) & (PVS.proba!=0.5)])

# |%%--%%| <7xlGswNfjf|qOnJlJxT8N>

sns.histplot(data=PVS,x='pvalue',hue="proba")

# |%%--%%| <qOnJlJxT8N|N2QGbzZ97s>

p_values=[]
pbs=[]
for i,s in enumerate(df.switch.values):
    vv=np.array(df.iloc[i].meanPos[s:])
    if(len(vv)>0):
        vv=vv[~np.isnan(vv)]
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vv)
        #print("Bias test statistic:", jarque_bera_stat)
        print("Bias p-value:", jarque_bera_p_value)
        p_values.append(jarque_bera_p_value)
        pbs.append(df.iloc[i].proba)
        plt.hist(vv,label="Pos Bias",alpha=.5)
    
    #Non Bias block
    vvv=np.array(df.iloc[i].meanPos[:s])
    if len(vvv)>0:
        vvv=vvv[~np.isnan(vvv)]
        plt.hist(vvv,label="Pos No Bias",alpha=.5)
        jarque_bera_stat, jarque_bera_p_value = stats.jarque_bera(vvv)
        
    #print("No Bias test statistic:", jarque_bera_stat)
        print("No Bias p-value:", jarque_bera_p_value)
    plt.title(df.iloc[i]["name"]+', '+str(df.iloc[i].proba))
    plt.legend()
    plt.show()

# |%%--%%| <N2QGbzZ97s|c4fx8sqyNT>

PVS=pd.DataFrame({'pvalue':p_values,"proba":pbs})
sns.boxplot(data=PVS,x="proba",y="pvalue")

# |%%--%%| <c4fx8sqyNT|pQGV16nCMI>

len(PVS[(PVS.pvalue<0.01)])

# |%%--%%| <pQGV16nCMI|Tx7aza1OvX>

df.tgt_direction

# |%%--%%| <Tx7aza1OvX|NUwKDwtTTK>

l=df[['proba','name','meanVeloBias','meanVeloNoBias','stdVeloBias','stdVeloNoBias', 'meanPosBias','meanPosNoBias','stdPosBias','stdPosNoBias']].groupby(["name","proba"]).mean()
l.reset_index(inplace=True)
l

# |%%--%%| <NUwKDwtTTK|1BlFFdoHOM>

sns.lmplot(data=l,x="proba",y="meanVeloBias")

# |%%--%%| <1BlFFdoHOM|tvz5apXgf3>

sns.lmplot(data=l,x="meanPosBias",y="stdPosBias",hue='proba',height=10)

# |%%--%%| <tvz5apXgf3|xEcy1SsTwZ>

sns.lmplot(data=l,x="meanVeloBias",y="stdVeloBias",hue='proba',height=10)
plt.savefig("corrVeloBiasAdults")

# |%%--%%| <xEcy1SsTwZ|GVxafDJ7OM>

sns.lmplot(data=l,x="meanPosBias",y="meanVeloBias",hue='proba',height=10)

# |%%--%%| <GVxafDJ7OM|XKAlWe8YkL>

Slope=[]
for n in np.unique(df.name):
    
    slope, intercept, r_value, p_value, std_err =linregress(df[df.name==n].proba,df[df.name==n].meanPosBias)
    plt.figure(figsize=(12,6))
    plt.scatter(df[df.name==n].proba,df[df.name==n].meanPosBias)
    plt.title(n+", slope="+str('% 6.3f' % slope)+ " p="+str('% 6.3f' % p_value))
    plt.xlabel("Proba")
    plt.ylabel("meanPosBias Bias")

    #Getting the slope
    Slope.append(slope)

    plt.plot(df[df.name==n].proba,df[df.name==n].proba*slope+intercept,c="r",alpha=0.5)
    plt.show()

# |%%--%%| <XKAlWe8YkL|44NKm84z4g>

ss=np.array(Slope)
Slope=ss[~np.isnan(ss)]
s=l.loc[~l.name.isin(nonValidPosName)].groupby("name").mean().stdPosNoBias.values

# |%%--%%| <44NKm84z4g|RYXShcHuVR>

plt.scatter(s,Slope)
plt.xlabel("stdPosNoBias")
plt.ylabel('Slope')

# |%%--%%| <RYXShcHuVR|noutI8JPaL>

Slope=[]
for n in np.unique(df.name):
    
    slope, intercept, r_value, p_value, std_err =linregress(df[df.name==n].proba,df[df.name==n].meanVeloBias)
    plt.figure(figsize=(12,6))
    plt.scatter(df[df.name==n].proba,df[df.name==n].meanVeloBias)
    plt.title(n+", slope="+str('% 6.3f' % slope)+ " p="+str('% 6.3f' % p_value))
    plt.xlabel("Proba")
    plt.ylabel("meanVelo Bias")

    #Getting the slope
    Slope.append(slope)

    plt.plot(df[df.name==n].proba,df[df.name==n].proba*slope+intercept,c="r",alpha=0.5)
    plt.show()

# |%%--%%| <noutI8JPaL|UX26o8C6PU>

ss=np.array(Slope)

# |%%--%%| <UX26o8C6PU|an6HnPauks>

Slope=ss[~np.isnan(ss)]

# |%%--%%| <an6HnPauks|h02Ql0PANK>

s=l.loc[~l.name.isin(nonValidVeloName)].groupby("name").mean().stdVeloNoBias.values


# |%%--%%| <h02Ql0PANK|NNNvKuDRCW>

len(s)

# |%%--%%| <NNNvKuDRCW|7DOa2jVUVF>

plt.scatter(s,Slope)
plt.xlabel("stdVeloNoBias")
plt.ylabel('Slope')

# |%%--%%| <7DOa2jVUVF|Zr1owIPGkq>

import statsmodels.api as sm



data = {
    'X': s,
    'Y': Slope,
}

# Create a DataFrame
relBiasNoBias = pd.DataFrame(data)

# Add a constant (intercept) to the independent variable
X = sm.add_constant(relBiasNoBias['X'])

# Fit the linear regression model
model = sm.OLS(relBiasNoBias['Y'], X).fit()

# Get the R-squared (R²) value
r_squared = model.rsquared
print(f'R-squared (R²) value: {r_squared:.4f}')

# Get the p-value for the slope coefficient (X)
p_value = model.pvalues['X']
print(f'P-value for the slope coefficient: {p_value:.4f}')

# Print the regression summary
print(model.summary())


# |%%--%%| <Zr1owIPGkq|37jJJfhnvA>

# Get the regression results
results = model.get_robustcov_results(cov_type='HC3')

# Create a scatterplot
sns.scatterplot(x='X', y='Y', data=relBiasNoBias)

# Plot the regression line
sns.lineplot(x=relBiasNoBias['X'], y=model.fittedvalues, color='red', label='Regression Line')
X_values=relBiasNoBias['X']
confidence_interval = results.conf_int()
upper_bound = confidence_interval[:, 1]  # Access the upper bound
lower_bound = confidence_interval[:, 0]  # Access the lower bound
#plt.fill_between(X_values, lower_bound, upper_bound, color='blue', alpha=0.3, label='95% Confidence Interval')
plt.xlabel("stdVeloNoBias")
plt.ylabel('Slope')
plt.savefig("SlopeAdults")


# |%%--%%| <37jJJfhnvA|xV2VrPNnuS>

sns.boxplot(data=df,x='proba',y='meanPosBias')

# |%%--%%| <xV2VrPNnuS|8z8gxkHvJi>

sns.boxplot(data=df,x='proba',y='stdPosBias')

# |%%--%%| <8z8gxkHvJi|ney8BxQIQI>

sns.boxplot(data=df,x='proba',y='meanPosNoBias')

# |%%--%%| <ney8BxQIQI|g0Kf2U48nF>

sns.scatterplot(data=df,x='stdPosBias' ,y='meanPosBias',s=100,hue='proba')

# |%%--%%| <g0Kf2U48nF|irOLgGhZTw>

sns.scatterplot(data=df[df.proba==1],x='stdPosBias' ,y='meanPosBias',s=100,hue='proba')

# |%%--%%| <irOLgGhZTw|eQDlBVPD4Y>

sns.scatterplot(data=df[~df.name.isin(nonValidVeloName)],x='meanVeloBias' ,y='meanPosBias',s=100,hue='proba')
plt.title("Correlation between the means of Position & Velocity in [80ms,120ms]")
plt.savefig("CorrPosVeloBiasAdult")

# |%%--%%| <eQDlBVPD4Y|D4aHzqxieO>

slope, intercept, r_value, p_value, std_err =linregress(df[~df.name.isin(nonValidVeloName) ].meanVeloBias,df[~df.name.isin(nonValidPosName)].meanPosBias)
plt.figure(figsize=(12,6))
plt.scatter(df.meanVeloBias,df.meanPosBias)
plt.title("Correlation between Velo and Pos Bias block R="+str('% 6.3f' % r_value)+ " p="+str('% 6.3f' % p_value))
plt.xlabel("Velo")
plt.ylabel("Pos")

plt.plot(df.meanVeloBias,df.meanVeloBias*slope+intercept,c="r",alpha=0.5)

# |%%--%%| <D4aHzqxieO|UHA31tWiZw>

slope, intercept, r_value, p_value, std_err =linregress(df[~df.name.isin(nonValidVeloName) ].stdPosBias,df[~df.name.isin(nonValidVeloName) ].meanPosBias)
plt.figure(figsize=(12,6))
plt.scatter(df.stdPosBias,df.meanPosBias)
plt.title("Correlation between StdPos and meanPos No Bias block R="+str('% 6.3f' % r_value)+ " p="+str('% 6.3f' % p_value))
plt.xlabel("StdPos")
plt.ylabel("meanPos")

plt.plot(df.stdPosBias,df.stdPosBias*slope+intercept,c="r",alpha=0.5)

# |%%--%%| <UHA31tWiZw|ih2GgXif9P>

slope, intercept, r_value, p_value, std_err =linregress(df[~df.name.isin(nonValidVeloName) ].stdVeloBias,df[~df.name.isin(nonValidVeloName) ].meanVeloBias)
plt.figure(figsize=(12,6))
plt.scatter(df.stdVeloBias,df.meanVeloBias)
plt.title("Correlation between StdVelo and meanVelo Bias block R="+str('% 6.3f' % r_value)+ " p="+str('% 6.3f' % p_value))
plt.xlabel("StdVelo")
plt.ylabel("meanVelo")

plt.plot(df.stdVeloBias,df.stdVeloBias*slope+intercept,c="r",alpha=0.5)

# |%%--%%| <ih2GgXif9P|KrZOxX2QKZ>

sns.lmplot(data=df[~df.name.isin(nonValidVeloName)],x='meanVeloBias' ,y='stdVeloBias',height=10,hue='proba')
plt.title("Correlation between StdVelo and meanVelo Bias block")
plt.savefig('CorrStdMeanVeloBiasAdults')

# |%%--%%| <KrZOxX2QKZ|C2umbgiY8u>

sns.scatterplot(data=df[~df.name.isin(nonValidVeloName)],x='stdPosNoBias' ,y='meanPosNoBias',s=100,hue='proba')

# |%%--%%| <C2umbgiY8u|De4Xo4eJLL>

sns.scatterplot(data=df[~df.name.isin(nonValidVeloName)],x='stdVeloNoBias' ,y='meanVeloNoBias',s=100,hue='proba')

# |%%--%%| <De4Xo4eJLL|YlIy91Yzw9>

#sns.scatterplot(data=df,x='CVVeloBias' ,y='meanVeloBias',s=100,hue='proba')

# |%%--%%| <YlIy91Yzw9|ZYM5fYOj63>

sns.scatterplot(data=df[(df.proba==0.7) &(~df.name.isin(nonValidVeloName))],x='stdVeloBias' ,y='meanVeloBias',s=100,hue='proba')

# |%%--%%| <ZYM5fYOj63|dUwwgr8eZH>

sns.scatterplot(data=df[(df.proba==0.9) &(~df.name.isin(nonValidVeloName))],x='stdVeloBias' ,y='meanVeloBias',s=100,hue='proba')

# |%%--%%| <dUwwgr8eZH|KaqWVVE0Dd>

sns.scatterplot(data=df[(df.proba==1) &(~df.name.isin(nonValidVeloName))],x='stdVeloBias' ,y='meanVeloBias',s=100,hue='proba')

# |%%--%%| <KaqWVVE0Dd|4kvZxua6SK>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot your data or customize each subplot as needed
axs[0, 0].scatter(df[df.proba==.7]["meanPosBias"],df[df.proba==.7]["stdPosBias"],label='Bias')
axs[0, 0].scatter(df[df.proba==.7]["meanPosNoBias"],df[df.proba==.7]["stdPosNoBias"],label='No Bias')
axs[0, 0].set_ylabel('Proba=0.7')
axs[0, 0].set_xlabel('meanPosBias')
axs[0, 0].set_title('Position', pad=20)
axs[0, 1].scatter(df[df.proba==.7].meanVeloBias,df[df.proba==.7].stdVeloBias,label='Bias')
axs[0, 1].scatter(df[df.proba==.7].meanVeloNoBias,df[df.proba==.7].stdVeloNoBias,label=' NoBias')

axs[0, 1].set_title('Velocity', pad=20)
axs[0, 1].set_xlabel('meanVeloBias')
axs[0, 1].set_ylabel('stdVeloBias')
axs[1, 0].scatter(df[df.proba==.9].meanPosBias,df[df.proba==.9].stdPosBias,label='Bias')
axs[1, 0].scatter(df[df.proba==.9].meanPosNoBias,df[df.proba==.9].stdPosNoBias,label='No Bias')

axs[1, 0].set_ylabel('Proba=0.9')
axs[1, 1].scatter(df[df.proba==.9].meanVeloBias,df[df.proba==.9].stdVeloBias,label='Bias')
axs[1, 1].scatter(df[df.proba==.9].meanVeloNoBias,df[df.proba==.9].stdVeloNoBias,label='No Bias')
axs[1, 1].set_ylabel('stdVeloBias')


axs[2, 0].scatter(df[df.proba==1].meanPosBias,df[df.proba==1].stdPosBias,label='Bias')
axs[2, 0].scatter(df[df.proba==1].meanPosNoBias,df[df.proba==1].stdPosNoBias,label='No Bias')
axs[2, 0].set_ylabel('Proba=1')

axs[2, 1].scatter(df[df.proba==1].meanVeloBias,df[df.proba==1].stdVeloBias,label='Bias')
axs[2, 1].scatter(df[df.proba==1].meanVeloNoBias,df[df.proba==1].stdVeloNoBias,label='No Bias')
fig.suptitle('Correlation between the mean and the standard deviation\n of Velocity & Position for different Probabilities ',fontsize=20)

# Add legends to each subplot
for ax in axs.flatten():
    ax.legend()

fig.tight_layout()
plt.savefig("CorrstdMeanProbasAdult")

# |%%--%%| <4kvZxua6SK|6J2GydAoJS>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# Plot your data or customize each subplot as needed
axs[0, 0].scatter(df[df.proba==.7]["meanPosBias"],df[df.proba==.7]["stdPosBias"])
axs[0, 0].set_ylabel('Proba=0.7')
axs[0, 0].set_xlabel('meanPosBias')
axs[0, 0].set_title('Position', pad=20)
axs[0, 1].scatter(df[df.proba==.7].meanVeloBias,df[df.proba==.7].stdVeloBias)
axs[0, 1].set_title('Velocity', pad=20)
axs[0, 1].set_xlabel('meanVeloBias')
axs[0, 1].set_ylabel('stdVeloBias')
axs[1, 0].scatter(df[df.proba==.9].meanPosBias,df[df.proba==.9].stdPosBias)
axs[1, 0].set_ylabel('Proba=0.9')
axs[1, 1].scatter(df[df.proba==.9].meanVeloBias,df[df.proba==.9].stdVeloBias)
axs[2, 0].scatter(df[df.proba==1].meanPosBias,df[df.proba==1].stdPosBias)
axs[2, 0].set_ylabel('Proba=1')
axs[2, 1].scatter(df[df.proba==1].meanVeloBias,df[df.proba==1].stdVeloBias)
fig.suptitle('Correlation between the mean and the standard deviation',fontsize=20)
fig.tight_layout()

# |%%--%%| <6J2GydAoJS|kjlNxi4Cin>

# Create a figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
#put the hue for the subjects
# Plot your data or customize each subplot as needed
slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==.7)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.7) & (df.stdPosBias<=20)]["stdPosBias"])
axs[0, 0].scatter(df[(df.proba==.7)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.7) & (df.stdPosBias<=20)]["stdPosBias"])
axs[0, 0].plot(df[(df.proba==.7)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.7)& (df.stdPosBias<=20)]["meanPosBias"]*slope+intercept,c="r",alpha=0.5)
axs[0, 0].set_ylabel('Proba=0.7')
axs[0, 0].set_xlabel('meanPosBias')
axs[0, 0].set_title('Position'+' p='+str('% 6.3f' % p_value), pad=20)

slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==.7)& (df.stdVeloBias<=20)]["meanVeloBias"],df[(df.proba==.7) & (df.stdVeloBias<=20)]["stdVeloBias"])
axs[0, 1].scatter(df[(df.proba==.7)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==.7) & (df.stdPosBias<=20)]["stdVeloBias"])
axs[0, 1].plot(df[(df.proba==.7)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==.7)& (df.stdPosBias<=20)]["meanVeloBias"]*slope+intercept,c="r",alpha=0.5)
axs[0, 1].set_ylabel('stdVeloBias')
axs[0, 1].set_xlabel('meanVeloBias')
axs[0, 1].set_title('Velo'+' p='+str('% 6.3f' % p_value), pad=20)

slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==.9)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.9) & (df.stdPosBias<=20)]["stdPosBias"])
axs[1, 0].scatter(df[(df.proba==.9)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.9) & (df.stdPosBias<=20)]["stdPosBias"])
axs[1, 0].plot(df[(df.proba==.9)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==.9)& (df.stdPosBias<=20)]["meanPosBias"]*slope+intercept,c="r",alpha=0.5)
axs[1, 0].set_ylabel('Proba=0.9')
axs[1, 0].set_xlabel('meanPosBias')
axs[1, 0].set_title('Position'+' p='+str('% 6.3f' % p_value), pad=20)

slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==.9)& (df.stdVeloBias<=20)]["meanVeloBias"],df[(df.proba==.9) & (df.stdVeloBias<=20)]["stdVeloBias"])
axs[1, 1].scatter(df[(df.proba==.9)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==.9) & (df.stdPosBias<=20)]["stdVeloBias"])
axs[1, 1].plot(df[(df.proba==.9)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==.9)& (df.stdPosBias<=20)]["meanVeloBias"]*slope+intercept,c="r",alpha=0.5)
axs[1, 1].set_ylabel('stdVeloBias')
axs[1, 1].set_xlabel('meanVeloBias')
axs[1, 1].set_title('Velo'+' p='+str('% 6.3f' % p_value), pad=20)


slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==1)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==1) & (df.stdPosBias<=20)]["stdPosBias"])
axs[2, 0].scatter(df[(df.proba==1)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==1) & (df.stdPosBias<=20)]["stdPosBias"])
axs[2, 0].plot(df[(df.proba==1)& (df.stdPosBias<=20)]["meanPosBias"],df[(df.proba==1)& (df.stdPosBias<=20)]["meanPosBias"]*slope+intercept,c="r",alpha=0.5)
axs[2, 0].set_ylabel('Proba=1')
axs[2, 0].set_xlabel('meanPosBias')
axs[2, 0].set_title('Position'+' p='+str('% 6.3f' % p_value), pad=20)

slope, intercept, r_value, p_value, std_err =linregress(df[(df.proba==1)& (df.stdVeloBias<=20)]["meanVeloBias"],df[(df.proba==1) & (df.stdVeloBias<=20)]["stdVeloBias"])
axs[2, 1].scatter(df[(df.proba==1)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==1) & (df.stdPosBias<=20)]["stdVeloBias"])
axs[2, 1].plot(df[(df.proba==1)& (df.stdPosBias<=20)]["meanVeloBias"],df[(df.proba==1)& (df.stdPosBias<=20)]["meanVeloBias"]*slope+intercept,c="r",alpha=0.5)
axs[2, 1].set_ylabel('stdVeloBias')
axs[2, 1].set_xlabel('meanVeloBias')
axs[2, 1].set_title('Velo'+' p='+str('% 6.3f' % p_value), pad=20)


fig.suptitle('Correlation between the mean and the standard deviation\n Bias Block  ',fontsize=20)
fig.tight_layout()
plt.savefig("CorrstdMeanProbasAdultsBias")

# |%%--%%| <kjlNxi4Cin|eiuKp5AAhY>

sns.boxplot(data=df,x='proba',y='meanVeloBias')

# |%%--%%| <eiuKp5AAhY|Ti0A4pk3Xy>

sns.boxplot(data=df,x='proba',y='stdVeloBias')

# |%%--%%| <Ti0A4pk3Xy|37KMHHFX5y>

sns.boxplot(data=df,x='proba',y='stdPosBias')


# |%%--%%| <37KMHHFX5y|aJeCFZwNFA>


sns.boxplot(data=df,x='proba',y='stdPosNoBias')


# |%%--%%| <aJeCFZwNFA|hbt41XthID>

sns.boxplot(data=df,x='proba',y='stdVeloBias')


# |%%--%%| <hbt41XthID|kTztaqxRE1>

sns.boxplot(data=df,x='proba',y='stdVeloNoBias')

# |%%--%%| <kTztaqxRE1|VsTEI6Lvk8>
r"""°°°
## Does the saccade comes to compensate in the opposite motion of anticipation
°°°"""
# |%%--%%| <VsTEI6Lvk8|DVyNzKB4ND>

df.head()

# |%%--%%| <DVyNzKB4ND|eiZrjdAo2N>

df_filtered = df[df['SaccDirection'].apply(len) > 0][['meanVelo','SaccDirection','SaccTrial','tgt_direction']]
df_filtered

# |%%--%%| <eiZrjdAo2N|EBzMAFR9Fy>

def extract_values(row):
    sacc_trial_indices = [int(idx) - 1 for idx in row['SaccTrial']]
    
    meanVelo_values = []
    tgt_direction_values = []

    # Iterate through the indices and append the values
    for idx in sacc_trial_indices:
        print(len(row['meanVelo']))
        if 0 <= idx < len(row['meanVelo']):
            meanVelo_values.append(row['meanVelo'][idx])
            tgt_direction_values.append(row['tgt_direction'][idx])
        else:
            print(f"Invalid index {idx} for meanVelo")


# |%%--%%| <EBzMAFR9Fy|iZGtv6p8bf>

df_filtered.apply(extract_values, axis=1)

# |%%--%%| <iZGtv6p8bf|GggEb5qjS9>



# |%%--%%| <GggEb5qjS9|BV66xJ1JmB>



# |%%--%%| <BV66xJ1JmB|Vs4CbsWQ8E>

df.SaccDirection.values[12]

# |%%--%%| <Vs4CbsWQ8E|AUiZQrXcGd>

np.take(df.meanVelo.values[12],np.array(df.SaccTrial.values[12]-1,dtype='int'))

# |%%--%%| <AUiZQrXcGd|3RBo9RRDf8>

excluded_names=df[df['meanVelo'].apply(lambda x: len(x)) != 150].name.values
excluded_names

# |%%--%%| <3RBo9RRDf8|Ir1BPSAriq>

df_prime=df[~df['name'].isin(excluded_names)]

# |%%--%%| <Ir1BPSAriq|AXVuK82MEU>

for i in range(len(df_prime)):
    
    plt.scatter(df_prime.SaccDirection.values[i], np.take(df_prime.meanVelo.values[i],np.array(df_prime.SaccTrial.values[i]-1,dtype='int')))
    

# |%%--%%| <AXVuK82MEU|qgHnroNhfJ>



concatenated_values = []
for i in range(len(df_prime)):
    values = np.take(df_prime.meanVelo.values[i], np.array(df_prime.SaccTrial.values[i]-1, dtype='int'))
    concatenated_values.extend(values)

result_df = pd.DataFrame({'Velo': concatenated_values})
result_df["SaccD"]=np.concatenate(df_prime.SaccDirection.values)
sns.boxplot(data=result_df,x="SaccD",y='Velo')

# |%%--%%| <qgHnroNhfJ|ayMwuZtwvf>

Sacc[Sacc.eye=='R']

# |%%--%%| <ayMwuZtwvf|TByk0g1h86>

plt.hist(Sacc[Sacc.eye=='R'].stime)

# |%%--%%| <TByk0g1h86|yhbAhylwh2>

len(Sacc[Sacc.eye=='R'].stime)

# |%%--%%| <yhbAhylwh2|tIVtRI4mom>

len(df.tgt_direction[0])

# |%%--%%| <tIVtRI4mom|XtpH2mxADI>

from scipy import stats

# |%%--%%| <XtpH2mxADI|cU4qDUNiQB>

result_df.dropna(inplace=True)

# |%%--%%| <cU4qDUNiQB|NWDFsJynw3>

# Perform independent two-sample t-test
t_statistic, p_value = stats.ttest_ind(result_df[result_df.SaccD==1].Velo, result_df[result_df.SaccD==-1].Velo,equal_var=False)

# Check the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)


# |%%--%%| <NWDFsJynw3|ZfYze4sSSq>

result_df[result_df.SaccD==-1].Velo

# |%%--%%| <ZfYze4sSSq|LYYbjYLxb4>
r"""°°°
### Xarray part
°°°"""
# |%%--%%| <LYYbjYLxb4|BIRYeV6FvD>

import xarray as xr

# Get the maximum number of trials
max_trials = df["meanPos"].apply(len).max()

# Create empty arrays to store the data
mean_pos_array = np.zeros((len(df), max_trials))
std_pos_array = np.zeros((len(df), max_trials))
mean_velo_array = np.zeros((len(df), max_trials))
std_velo_array = np.zeros((len(df), max_trials))


# Populate the arrays with data from the DataFrame
for i, (_, row) in enumerate(df.iterrows()):
    mean_pos_array[i, :len(row["meanPos"])] = row["meanPos"]
    std_pos_array[i, :len(row["stdPos"])] = row["stdPos"]
    mean_velo_array[i, :len(row["meanVelo"])] = row["meanVelo"]
    std_velo_array[i, :len(row["stdVelo"])] = row["stdVelo"]

# Create the xarray Dataset
ds = xr.Dataset(
    {
        "meanPos": (["scenario", "trials"], mean_pos_array),
        "stdPos": (["scenario", "trials"], std_pos_array),
        "meanVelo": (["scenario", "trials"], mean_velo_array),
        "stdVelo": (["scenario", "trials"], std_velo_array),
    },
    coords={"proba": np.unique(df.proba), "name":np.unique(df.name), "trials": range(max_trials)},
)


# Print the xarray dataset
print(ds)



# |%%--%%| <BIRYeV6FvD|ilmWPPL7f7>

df

# |%%--%%| <ilmWPPL7f7|N1y3bk62ms>

ds.sel(proba=0.9)

# |%%--%%| <N1y3bk62ms|xwzWbaUPX4>



# |%%--%%| <xwzWbaUPX4|1kGwabsuiG>

ds.keys()

# |%%--%%| <1kGwabsuiG|mjdvKeqKXH>

import xarray as xr
import numpy as np

# Set proba and name as dimensions in the DataFrame
df = df.set_index(['proba', 'name'])

# Get the maximum number of trials
max_trials = df["meanPos"].apply(len).max()

# Create empty arrays to store the data
mean_pos_array = np.zeros((len(df), max_trials))
std_pos_array = np.zeros((len(df), max_trials))
mean_velo_array = np.zeros((len(df), max_trials))
std_velo_array = np.zeros((len(df), max_trials))

# Create the xarray Dataset
ds = xr.Dataset()

# Populate the arrays with data from the DataFrame
for _, row in df.iterrows():
    proba, name = row.name
    trials = row['meanPos']

    scenario = f"proba={proba}, name={name}"
    
    # Pad the trials with NaN values if it has fewer than max_trials
    trials_padded = np.pad(trials, (0, max_trials - len(trials)), mode='constant', constant_values=np.nan)
    
    # Assign the data to the xarray Dataset
    ds = ds.assign(
        {
            f"meanPos_{scenario}": (["trials"], trials_padded),
            f"stdPos_{scenario}": (["trials"], np.pad(row["stdPos"], (0, max_trials - len(row["stdPos"])), mode='constant', constant_values=np.nan)),
            f"meanVelo_{scenario}": (["trials"], np.pad(row["meanVelo"], (0, max_trials - len(row["meanVelo"])), mode='constant', constant_values=np.nan)),
            f"stdVelo_{scenario}": (["trials"], np.pad(row["stdVelo"], (0, max_trials - len(row["stdVelo"])), mode='constant', constant_values=np.nan)),
        }
    )

    # Add the scenario as a coordinate
    ds.coords[scenario] = 1

# Add the trials dimension
ds.coords["trials"] = range(max_trials)

# Print the xarray dataset
print(ds)


# |%%--%%| <mjdvKeqKXH|OKI8GR5U08>



# |%%--%%| <OKI8GR5U08|Igp75ugVXV>



# |%%--%%| <Igp75ugVXV|X4CeXPselv>



# |%%--%%| <X4CeXPselv|N1pWktOAAc>



# |%%--%%| <N1pWktOAAc|RSXyKyWz7c>



# |%%--%%| <RSXyKyWz7c|IS3i5wQ8EH>



# |%%--%%| <IS3i5wQ8EH|k1DkCMtOtE>



# |%%--%%| <k1DkCMtOtE|5C8JDjO3AF>



# |%%--%%| <5C8JDjO3AF|fuzc9B8tM7>



# |%%--%%| <fuzc9B8tM7|OoIWVkqs6A>



# |%%--%%| <OoIWVkqs6A|FaaU0xVGqe>

Slope=[]
for n in np.unique(df.name):
    
    slope, intercept, r_value, p_value, std_err =linregress(df[df.name==n].proba,df[df.name==n].meanVeloBias)
    plt.figure(figsize=(12,6))
    plt.scatter(df[df.name==n].proba,df[df.name==n].meanVeloBias)
    plt.title(n+", slope="+str('% 6.3f' % slope)+ " p="+str('% 6.3f' % p_value))
    plt.xlabel("Proba")
    plt.ylabel("meanVelo Bias")

    #Getting the slope
    Slope.append(slope)

    plt.plot(df[df.name==n].proba,df[df.name==n].proba*slope+intercept,c="r",alpha=0.5)
    plt.show()
