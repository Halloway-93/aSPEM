# %%
import BIDSification_eyetrackingData as BIDS

# %%
path_oldData = '/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/' # Path of the data directory to BIDSified

# %%
"""
# InfoFiles
"""

# %%
"""
Create a **TSV** file containing information about the data files in the data directory to be BIDSified

It **MUST** contain the columns:

- `filename` - *name of the data files*
- `filepath` - *path of the data files to the directory. If they are in the main directory:* `filepath=''`
- `participant_id` - *participant identifier*


It **may also** contain the columns:

- `eventsfilename` - *name of the tsv file associated with the data which gathers information about each trial (e.g. the colour of the target if it is different for each trial)*
- `ses` - *Name of the Session*
- `task` - *Name of the Task*
- `acq` - *Name of the Aquisition*
- `run` - *IndexRun*
- *information about the participant (*`age`*,* `QI`*,* `laterality`*, ...)*
- *task information if it is valid for all trials (*`Proba`*,* `speedTarget`*,* `colorTarget`*, ...)*
"""

# %%
"""
To create this file automatically, fill in the participant_id column automatically but also check if it is complete you can use the class:
"""

# %%
print(BIDS.StandardisationProcess.__doc__)

# %%
print(BIDS.StandardisationProcess.create_infoFiles.__doc__)

# %%
process = BIDS.StandardisationProcess(dirpath=path_oldData)

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------
dataformat = '.asc' # Data format

#------------------------------------------------------------------------------
# to apply the function:
#------------------------------------------------------------------------------
process.create_infoFiles(dataformat=dataformat)

# %%
"""
**You must then open this file and complete it.**

**to modify the file**

you need to install pandas, open a terminal and run the command:
- try: `pip install pandas`
- or else : `pip3 install pandas`
"""

# %%
import pandas as pd

# %%
infoFiles = pd.read_csv('/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/infoFiles.tsv', delimiter=' ') # open
# %%
infoFiles
infoFiles.ses
# %%
for name in infoFiles.filename:
    infoFiles.loc[infoFiles.filename==name, 'participant_id'] = "{:03d}".format(int(name[1]))
    infoFiles.loc[infoFiles.filename==name, 'ses'] = "{:03d}".format(int(name.split('.')[0][-1]))


# %%
infoFiles

# %%
infoFiles.to_csv('/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles/infoFiles.tsv', sep=' ') # save

# %%
"""
You must then open this file and complete it.
"""

# %%
"""
# Setting File
"""

# %%
"""
Create a **JSON** file containing the general parameters of the task in the data directory to be BIDSified.
"""

# %%
process = BIDS.StandardisationProcess(dirpath=path_oldData)

#------------------------------------------------------------------------------
# to apply the function:
#------------------------------------------------------------------------------
process.create_settingsFile()

# %%
"""
**You must then open this file and complete it.**

You MUST complete the following keys:

- **EnvironmentCoordinates** - Coordinates origin (or zero), for gaze-on-screen coordinates, this can be for example: `"top-left"` or `"center"`. (string)
- **ScreenSize** - Screen size in cm (for example `[47.2, 29.5]`) (array of numbers)
- **ScreenDistance** - Distance between the participant's eye and the screen. in cm (number)
- **SamplingFrequency** **\*** - Sampling frequency (in Hz) (number)
- **SampleCoordinateUnit** **\*** - Unit of individual samples (`"pixel"`, `"mm"` or `"cm"`) (string)
- **SampleCoordinateSystem** **\*** - Classical screen-based eye tracking data would be `"gaze-on-screen"` (string)
- **ScreenResolution** **\*** - Screen resolution in pixel (array of integers)

**\*** *indicates that the variable will be automatically extracted from the data if it comes from an eyelink file*
"""

# %%
settings = pd.read_json('/Users/hamzahalloway/Nextcloud/Shared/HAMZA_PhD/Data/Probant_DevAsd/DATA/Controles//settings.json', orient='index').T # open

# %%
settings

# %%
settings.TaskName = "rdkdir"
settings.EnvironmentCoordinates = "top-left"
settings.ScreenSize = [[70, 40]]
settings.ScreenDistance = 57

# %%
settings

# %%
settings.T[0].to_json('../Data_ASC/settings.json', orient='index', indent=4) # save

# %%
"""
# datasetdescriptionfilename
"""

# %%
process.create_dataset_description()

# %%
dataset_description = pd.read_json('../Data_ASC/dataset_description.json', orient='index').T # open

# %%
dataset_description

# %%
dataset_description.Name = "rdkdir"

# %%
dataset_description

# %%
dataset_description.T[0].to_json('../Data_ASC/dataset_description.json', orient='index', indent=4) # save

# %%
"""
# settingsEvents
"""

# %%
process.create_settingsEvents('../Data_ASC/infoFiles.tsv')

# %%
settingsEvents = pd.read_json('../Data_ASC/settingsEvents.json', orient='index').T # open

# %%
settingsEvents

# %%
settingsEvents['resp.coh'] = 'a'
settingsEvents['resp.dirchoice'] = 'a'
settingsEvents['resp.rdkDir'] = 'a'
settingsEvents['resp.trialIdx'] = 'a'

# %%
settingsEvents

# %%
"""
# Data standardisation
"""

# %%
print(BIDS.DataStandardisation.__doc__)

# %%
#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

path_newData = '../Data_BIDS' # Path of the new BIDS data directory

# Name of the file containing the information on the files to be BIDSified
infofilesname = 'infoFiles.tsv'
# Name of the file containing the information on the files to be BIDSified
settingsfilename = 'settings.json'
# Name of the file containing the events settings in the BIDSified data
settingsEventsfilename = 'settingsEvents.json'
# Name of the file describing the dataset
datasetdescriptionfilename = 'dataset_description.json'

eyetracktype = 'Eyelink' # Name of the type of eyetackeur used
dataformat = '.asc' # Data format

# List of events to be extracted from the trials
saved_events = {"fixationOn": {"Description": "appearance of the fixation point"},
                "fixationOff": {"Description": "disappearance of the fixation point"},
                "rdkOn": {"Description": "appearance of the moving target"},
                "rdkOff": {"Description": "disappearance of the moving target"}}


StartMessage='Trialinfo' # Message marking the start of the trial
EndMessage= None # Message marking the end of the trial



#------------------------------------------------------------------------------
# to apply the function:
#------------------------------------------------------------------------------
BIDS.DataStandardisation(path_oldData=path_oldData,
                         path_newData=path_newData,
                         infofilesname=infofilesname,
                         settingsfilename=settingsfilename,
                         settingsEventsfilename=settingsEventsfilename,
                         datasetdescriptionfilename = datasetdescriptionfilename,
                         eyetracktype=eyetracktype,
                         dataformat=dataformat,
                         saved_events=saved_events,
                         StartMessage=StartMessage,
                         EndMessage=EndMessage);

# %%


# %%
