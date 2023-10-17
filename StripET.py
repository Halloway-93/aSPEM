
from __future__ import division
from __future__ import print_function

import pylink
import os
import platform
import random
import time
import sys
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import visual, core, event, monitors, gui
from math import pi
from string import ascii_letters, digits
import random
from scipy import stats
import pyglet
import pylink
from screeninfo import get_monitors

# Switch to the script folder
script_path = os.path.dirname(sys.argv[0])
if len(script_path) != 0:
    os.chdir(script_path)

# Show only critical log message in the PsychoPy console
from psychopy import logging
logging.console.setLevel(logging.CRITICAL)

# Set this variable to True if you use the built-in retina screen as your
# primary display device on macOS. If have an external monitor, set this
# variable True if you   choose to "Optimize for Built-in Retina Display"
# in the Displays preference settings.
use_retina = False

# Set this variable to True to run the script in "Dummy Mode"
dummy_mode = False

# Set this variable to True to run the task in full screen mode
# It is easier to debug the script in non-fullscreen mode
full_screen = True

# Set up EDF data file name and local data folder
#
# The EDF data filename should not exceed 8 alphanumeric characters
# use ONLY number 0-9, letters, & _ (underscore) in the filename
edf_fname = 'TEST'

# Prompt user to specify an EDF data filename
# before we open a fullscreen window
dlg_title = 'Enter EDF File Name'
dlg_prompt = 'Please enter a file name with 8 or fewer characters\n' + \
             '[letters, numbers, and underscore].'

# loop until we get a valid filename
while True:
    dlg = gui.Dlg(dlg_title)
    dlg.addText(dlg_prompt)
    dlg.addField('File Name:', edf_fname)
    # show dialog and wait for OK or Cancel
    ok_data = dlg.show()
    if dlg.OK:  # if ok_data is not None
        print('EDF data filename: {}'.format(ok_data[0]))
    else:
        print('user cancelled')
        core.quit()
        sys.exit()

    # get the string entered by the experimenter
    tmp_str = dlg.data[0]
    # strip trailing characters, ignore the ".edf" extension
    edf_fname = tmp_str.rstrip().split('.')[0]

    # check if the filename is valid (length <= 8 & no special char)
    allowed_char = ascii_letters + digits + '_'
    if not all([c in allowed_char for c in edf_fname]):
        print('ERROR: Invalid EDF filename')
    elif len(edf_fname) > 8:
        print('ERROR: EDF filename should not exceed 8 characters')
    else:
        break



# Set up a folder to store the EDF data files and the associated resources
# e.g., files defining the interest areas used in each trial
results_folder = 'results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time
time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = edf_fname + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(results_folder, session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)

# Step 1: Connect to the EyeLink Host PC
#
# The Host IP address, by default, is "100.1.1.1".
# the "el_tracker" objected created here can be accessed through the Pylink
# Set the Host PC address to "None" (without quotes) to run the script
# in "Dummy Mode"
if dummy_mode:
    el_tracker = pylink.EyeLink(None)
else:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()

# Step 2: Open an EDF data file on the Host PC
edf_file = edf_fname + ".EDF"
try:
    el_tracker.openDataFile(edf_file)
except RuntimeError as err:
    print('ERROR:', err)
    # close the link if we have one open
    if el_tracker.isConnected():
        el_tracker.close()
    core.quit()
    sys.exit()

# Add a header text to the EDF file to identify the current experiment name
# This is OPTIONAL. If your text starts with "RECORDED BY " it will be
# available in DataViewer's Inspector window by clicking
# the EDF session node in the top panel and looking for the "Recorded By:"
# field in the bottom panel of the Inspector.
preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)

# Step 3: Configure the tracker
#
# Put the tracker in offline mode before we change tracking parameters
el_tracker.setOfflineMode()

# Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
# 5-EyeLink 1000 Plus, 6-Portable DUO
eyelink_ver = 0  # set version to 0, in case running in Dummy mode
if not dummy_mode:
    vstr = el_tracker.getTrackerVersionString()
    eyelink_ver = int(vstr.split()[-1].split('.')[0])
    # print out some version info in the shell
    print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

# File and Link data control
# what eye events to save in the EDF file, include everything by default
file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
# what eye events to make available over the link, include everything by default
link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
# what sample data to save in the EDF data file and to make available
# over the link, include the 'HTARGET' flag to save head target sticker
# data for supported eye trackers
if eyelink_ver > 3:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
else:
    file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
    link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

# Optional tracking parameters
# Sample rate, 250, 500, 1000, or 2000, check your tracker specification
# if eyelink_ver > 2:
#     el_tracker.sendCommand("sample_rate 1000")

# Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
el_tracker.sendCommand("calibration_type = HV9")
# Set a gamepad button to accept calibration/drift check target
# You need a supported gamepad/button box that is connected to the Host PC
el_tracker.sendCommand("button_function 5 'accept_target_fixation'")

# Step 4: set up a graphics environment for calibration
#
# Get a list of all connected displays
all_screens = get_monitors()

# Specify the index for your secondary display
display_index = 1

# Check if the specified index is valid
if display_index < len(all_screens):
    # Get the screen properties for the specified display
    secondary_screen = all_screens[display_index]
    scn_x=secondary_screen.x
    scn_y=secondary_screen.y
else:
    display_index=0

# Open a window, be sure to specify monitor parameters
mon = monitors.Monitor('myMonitor', width=53.0, distance=70.0)
win = visual.Window(fullscr=full_screen,
                    monitor=mon,
                    winType='pyglet',
                    units='pix',screen=display_index)
if display_index!=0:
    # Get the Pyglet window object associated with the Psychopy window
    pyglet_win = win.winHandle
    pyglet_win.set_location(scn_x,-scn_y)

# get the native screen resolution used by PsychoPy
scn_width, scn_height = win.size
# resolution fix for Mac retina displays
if 'Darwin' in platform.system():
    if use_retina:
        scn_width = int(scn_width/2.0)
        scn_height = int(scn_height/2.0)

# Pass the display pixel coordinates (left, top, right, bottom) to the tracker
# see the EyeLink Installation Guide, "Customizing Screen Settings"
el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendCommand(el_coords)

# Write a DISPLAY_COORDS message to the EDF file
# Data Viewer needs this piece of info for proper visualization, see Data
# Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
el_tracker.sendMessage(dv_coords)

# Configure a graphics environment (genv) for tracker calibration
genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
print(genv)  # print out the version number of the CoreGraphics library

# Set background and foreground colors for the calibration target
# in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
foreground_color = (-1, -1, -1)
background_color = win.color
genv.setCalibrationColors(foreground_color, background_color)

# Use the default calibration target ('circle')
genv.setTargetType('picture')
genv.setPictureTarget(os.path.join('picture',"images","fixTarget.bmp"))

# Configure the size of the calibration target (in pixels)
# this option applies only to "circle" and "spiral" targets
#genv.setTargetSize(24)

# Beeps to play during calibration, validation and drift correction
# parameters: target, good, error
#     target -- sound to play when target moves
#     good -- sound to play on successful operation
#     error -- sound to play on failure or interruption
# Each parameter could be ''--default sound, 'off'--no sound, or a wav file
genv.setCalibrationSounds('', '', '')

# resolution fix for macOS retina display issues
if use_retina:
    genv.fixMacRetinaDisplay()

# Request Pylink to use the PsychoPy window we opened above for calibration
pylink.openGraphicsEx(genv)


# define a few helper functions for trial handling


def clear_screen(win):
    """ clear up the PsychoPy window"""

    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_msg(win, text, wait_for_keypress=True):
    """ Show task instructions on screen"""

    msg = visual.TextStim(win, text,
                          color=genv.getForegroundColor(),
                          wrapWidth=scn_width/2)
    clear_screen(win)
    msg.draw()
    win.flip()

    # wait indefinitely, terminates upon any key press
    if wait_for_keypress:
        event.waitKeys()
        clear_screen(win)


def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """

    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        # Put tracker in Offline mode
        el_tracker.setOfflineMode()

        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)

        # Close the edf data file on the Host
        el_tracker.closeDataFile()

        # Show a file transfer message on the screen
        msg = 'EDF data is transferring from EyeLink Host PC...'
        show_msg(win, msg, wait_for_keypress=False)

        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

        # Close the link to the tracker.
        el_tracker.close()

    # close the PsychoPy window
    win.close()

    # quit PsychoPy
    core.quit()
    sys.exit()


def abort_trial():
    """Ends recording """

    el_tracker = pylink.getEYELINK()

    # Stop recording
    if el_tracker.isRecording():
        # add 100 ms to catch final trial events
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    # clear the screen
    clear_screen(win)
    # Send a message to clear the Data Viewer screen
    #bgcolor_RGB = (116, 116, 116)
    #el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

    # send a message to mark trial end
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)

    return pylink.TRIAL_ERROR


def run_trial(numberOfTrials, trial_index,p_green=0.9,p_red=0.10):
    """ Helper function specifying the events that will occur in a single trial
    """

   
    # get a reference to the currently active EyeLink connection
    el_tracker = pylink.getEYELINK()

    # put the tracker in the offline mode first
    el_tracker.setOfflineMode()
    # clear the Host PC screen before we draw the backdrop
    el_tracker.sendCommand('clear_screen 0')
    # send a "TRIALID" message to mark the start of a trial, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    el_tracker.sendMessage('TRIALID %d' % trial_index)

    # record_status_message : show some info on the Host PC
    # here we show how many trial has been tested
    status_msg = 'TRIAL number %d' % trial_index
    el_tracker.sendCommand("record_status_message '%s'" % status_msg)

    # Step 6: Run a single trial
    
    # Set the number of dots
    num_dots = 1000

    # Define the size of the band in pixels
    band_width_px = win.size[0]
    band_height_px = 40

    # Create the fixation cross
    fixation_cross = visual.TextStim(win, text='+', color='white', height=25)

    # Create two circle buttons for color selection
    button_radius = 30
    button_pos=100 * (2 * (stats.bernoulli.rvs(0.5)) - 1)
    green_button = visual.Circle(win, radius=button_radius, pos=(0,button_pos), fillColor=(0,200,0))
    red_button = visual.Circle(win, radius=button_radius, pos=(0,-button_pos), fillColor=(200,0,0))

    
    if button_pos>0:
        print("green should be up")
    else:
        print("green should be down")


    #checking where the buttons are
    if button_pos>0:
        button_pos='up'
    else:
        button_pos='down'
    
    print("the button position is:",button_pos)

    
    while (not dummy_mode) & (trial_index == 5 or trial_index == 100 or trial_index==150 or trial_index==200): 
        # terminate the task if no longer connected to the tracker or
        # user pressed Ctrl-C to terminate the task
        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
            terminate_task()
            return pylink.ABORT_EXPT
        
        # drift-check and re-do the camera setup if ESC is pressed

        try:
            error=el_tracker.doDriftCorrect(int(scn_width/2.0), int(scn_height/2.0), 1, 1)
            
            #break following a successful drift-check
            if error is not pylink.ESC_KEY:
                break
        except:
            pass

    # put the tracker in the offline before starting recording
    el_tracker.setOfflineMode()

    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print("ERROR:", error)
        abort_trial()
        return pylink.TRIAL_ERROR

    # Allocate some time for the tracker to cache some samples
    pylink.pumpDelay(100)
    #clear the screen
    clear_screen(win)
    # Show the color selection buttons
    green_button.draw()
    red_button.draw()
    win.flip()
    el_tracker.sendMessage('cue_selection')
    
    # Wait for the user to press the upper arrow key ('up') or donw arrow key ('down')
    
    chosen_color = None
    while chosen_color not in ['up', 'down']:
        keys = event.waitKeys(keyList=['up', 'down'])
        if keys:
            chosen_color = keys[0]

    # Map the keypresses to color choices
    if chosen_color == button_pos:
        chosen_color = 'g'
    else:
        chosen_color = 'r'
    
    print("the chosen color is:",chosen_color)
    

    # Calculate the positions for the upper and lower bands
    # randomized vertical color position.
    vertDirection = 2 * (stats.bernoulli.rvs(0.5)) - 1
    green_band_y = vertDirection * int(win.size[1] / 8)
    red_band_y = -vertDirection * (int(win.size[1] / 8))

    print("the Y pos of the green band",green_band_y)
    print("the Y pos of the red band",red_band_y)
    



    # Initialize RDK stimuli coherence and direction with default values
    upper_rdk = visual.DotStim(win, nDots=num_dots, dotSize=5, fieldSize=(band_width_px, band_height_px), dotLife=30,
                            color=(0,1,0), coherence=0, dir=0, fieldPos=(0, green_band_y),speed=1)
    lower_rdk = visual.DotStim(win, nDots=num_dots, dotSize=5, fieldSize=(band_width_px, band_height_px), dotLife=30,
                            color=(1,0,0), coherence=0, dir=0, fieldPos=(0, red_band_y),speed=1)


    
    #green stimuli or #red stimuli
    #if chosen_color == 'g' or  chosen_color == 'r':

    p_green = 0.75  # Probability of the band going right
    dir_green = stats.bernoulli.rvs(p_green)
    if dir_green == 1:
        dir_green = 0
    else:
        dir_green = 180
    
    upper_rdk.dir = dir_green

#red stimuli
    p_red = 0.25  # Probability of the band going right
    dir_red = stats.bernoulli.rvs(p_red)
    if dir_red == 1:
        dir_red = 0
    else:
        dir_red = 180

    lower_rdk.dir = dir_red


    
    # Show the fixation cross for a random time between 500ms and 1000ms
    fixation_cross.draw()
    win.flip()
    el_tracker.sendMessage('fixation_cross')
    random_time_interval = random.uniform(0.5, 1.0)  # Random time between 500ms and 1000ms
    core.wait(random_time_interval)

    
    # Draw the RDK stimuli
    upper_rdk.draw()
    lower_rdk.draw()
    win.flip()
    el_tracker.sendMessage('TargetOnSet')
    #This waiting time will be replaced by the time of the end of the saccade+200ms
    #Getting the gaze position
    if not dummy_mode:
        #getting the eye position
        sample= el_tracker.getNextData()
        print("the sample is:",sample)
        #getting the Y position of the right eye
        rightY = sample.getRightEye().getGazeY()
        print("the gaze position is:",rightY)
        if chosen_color == 'g':
            while rightY > green_band_y +40 or rightY < green_band_y -40:
                sample = el_tracker.getNextData()
                rightY = sample.getRightEye().getGazeY()
                core.wait(0.001)
        else:
            while rightY < red_band_y-40 or rightY > red_band_y+40:
                sample = el_tracker.getNextData()
                rightY = sample.getRightEye().getGazeY()
                core.wait(0.001)
        core.wait(0.2) #wait 200ms before moving
    else:
        core.wait(1)#Wait 1000 ms before moving 



    # Animation loop for each trial
    timer = core.CountdownTimer(2.0)  # Set timer for 2000ms (2 second)

    while timer.getTime() > 0:
    
        # Reset the coherence for both RDK stimuli at the beginning of each trial
        upper_rdk.coherence = 1
        lower_rdk.coherence = 1
        # Draw the RDK stimuli
        upper_rdk.draw()
        lower_rdk.draw()
        # Update the window
        win.flip()

    # Send a message to clear the Data Viewer screen, get it ready for
    # drawing the pictures during visualization
    #bgcolor_RGB = (116, 116, 116)
    #el_tracker.sendMessage('!V CLEAR %d %d %d' % bgcolor_RGB)

  
    # abort the current trial if the tracker is no longer recording
    error = el_tracker.isRecording()
    if error is not pylink.TRIAL_OK:
        el_tracker.sendMessage('tracker_disconnected')
        abort_trial()
        return error

    # check keyboard events
    for keycode, modifier in event.getKeys(modifiers=True):
        # Abort a trial if "ESCAPE" is pressed
        if keycode == 'escape':
            el_tracker.sendMessage('trial_skipped_by_user')
            # clear the screen
            clear_screen(win)
            # abort trial
            abort_trial()
            return pylink.SKIP_TRIAL

        # Terminate the task if Ctrl-c
        if keycode == 'c' and (modifier['ctrl'] is True):
            el_tracker.sendMessage('terminated_by_user')
            terminate_task()
            return pylink.ABORT_EXPT

    

    # clear the screen
    clear_screen(win)
    el_tracker.sendMessage('blank_screen')
    # stop recording; add 100 msec to catch final events before stopping
    pylink.pumpDelay(100)
    el_tracker.stopRecording()


# Step 5: Set up the camera and calibrate the tracker
#
# Show the task instructions
task_msg = 'In the task, please select one color RED or GREEN with the UP or DOWN arrow\n' + \
    'of the keyboard, then make a saccade toward the strip with the selected color\n' + \
    'Follow the movment of the strip with your eyes.\n' + \
    'Try to balance your choice between colors \n'

if dummy_mode:
    task_msg = task_msg + '\nNow, Press ENTER to start the task'
else:
    task_msg = task_msg + '\nNow, Press ENTER twice to calibrate tracker'
show_msg(win, task_msg)

# skip this step if running the script in Dummy Mode
if not dummy_mode:
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()

# Step 6: Run the experiment trials
numberOfTrials=5
trial_index=1
for i in range(numberOfTrials):
        
    run_trial(numberOfTrials, trial_index,p_green=0.9,p_red=0.10)
    trial_index += 1

# Step 7: disconnect, download the EDF file, then terminate the task
terminate_task()
