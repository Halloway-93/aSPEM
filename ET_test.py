import random

import pylink
from scipy import stats

from psychopy import core, event, visual

# Create a window
win = visual.Window(
    [800, 600], color=(0.5, 0.5, 0.5), units="pix", monitor="testMonitor", fullscr=False
)

# Set the number of dots
num_dots = 1000

# Define the size of the band in pixels
band_width_px = win.size[0]
band_height_px = 40

# Create the fixation cross
fixation_cross = visual.TextStim(win, text="+", color="white", height=25)

# Create two circle buttons for color selection
button_radius = 30
green_button = visual.Circle(
    win, radius=button_radius, pos=(-100, 0), fillColor="green"
)
red_button = visual.Circle(win, radius=button_radius, pos=(100, 0), fillColor="red")

##############################################################################################

# Connect to the EyeLink eye tracker
eyelink_tracker = pylink.EyeLink()


# Set up the connection with the EyeLink eye tracker
eyelink_tracker.sendCommand("screen_pixel_coords = 0 0 %d %d" % tuple(win.size))
eyelink_tracker.sendCommand("calibration_type = HV9")
eyelink_tracker.sendCommand("generate_default_targets = YES")
pylink.openGraphics(win)

# Calibrate the eye tracker
eyelink_tracker.doTrackerSetup()


# Number of trials
Ntrials = 5

for trial in range(Ntrials):
    # Start recording eye data
    eyelink_tracker.startRecording(1, 1, 1, 1)

    # Show the color selection buttons
    green_button.draw()
    red_button.draw()
    win.flip()

    # Wait for the user to press the left arrow key ('left') or right arrow key ('right')
    chosen_color = None
    while chosen_color not in ["left", "right"]:
        keys = event.waitKeys(keyList=["left", "right"])
        if keys:
            chosen_color = keys[0]

    # Map the keypresses to color choices
    if chosen_color == "left":
        chosen_color = "g"
    elif chosen_color == "right":
        chosen_color = "r"

    # Calculate the positions for the upper and lower bands
    # randomized vertical color position.
    vertDirection = 2 * (stats.bernoulli.rvs(0.5)) - 1
    green_band_y = vertDirection * int(win.size[1] / 8)
    red_band_y = -vertDirection * (int(win.size[1] / 8))

    # Initialize RDK stimuli coherence and direction with default values
    upper_rdk = visual.DotStim(
        win,
        nDots=num_dots,
        dotSize=5,
        fieldSize=(band_width_px, band_height_px),
        dotLife=30,
        color="green",
        coherence=0,
        dir=0,
        fieldPos=(0, green_band_y),
    )

    lower_rdk = visual.DotStim(
        win,
        nDots=num_dots,
        dotSize=5,
        fieldSize=(band_width_px, band_height_px),
        dotLife=30,
        color="red",
        coherence=0,
        dir=0,
        fieldPos=(0, red_band_y),
    )

    # green stimuli or #red stimuli
    if chosen_color == "g" or chosen_color == "r":
        p_green = 0.75  # Probability of the band going right
        dir_green = stats.bernoulli.rvs(p_green)
        if dir_green == 1:
            dir_green = 0
        else:
            dir_green = 180

        upper_rdk.dir = dir_green

        # red stimuli
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
    random_time_interval = random.uniform(
        0.5, 1.0
    )  # Random time between 500ms and 1000ms
    core.wait(random_time_interval)

    # Draw the RDK stimuli
    upper_rdk.draw()
    lower_rdk.draw()
    win.flip()
    # This waiting time will be replaced by the time of the end of the saccade+200ms
    core.wait(1)  # Wait 1000 ms before moving

    # Animation loop for each trial
    timer = core.CountdownTimer(2.0)  # Set timer for 1000ms (2 second)

    while timer.getTime() > 0:
        elapsed_time = (
            core.getTime()
        )  # Get the time elapsed since the start of the loop

        # Reset the coherence for both RDK stimuli at the beginning of each trial
        upper_rdk.coherence = 1.0
        lower_rdk.coherence = 1.0
        # Draw the RDK stimuli
        upper_rdk.draw()
        lower_rdk.draw()
        # Update the window
        win.flip()

        # Check for the 'Escape' key press to exit the loop
        if "escape" in event.getKeys():
            break
    # Stop recording eye data
    eyelink_tracker.stopRecording()

# Save the recorded eye data to an EDF file
edf_filename = "participant_data.edf"  # Replace with your desired file name
pylink.msecDelay(500)  # Give EyeLink time to finish saving the data
eyelink_tracker.receiveDataFile(edf_filename)

# Close the EyeLink connection
eyelink_tracker.close()

# Close the window at the end of all trials
win.close()
core.quit()
