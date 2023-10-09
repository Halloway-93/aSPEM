from psychopy import visual, core, event
import random

win = visual.Window(size=(800, 600), fullscr=False, allowGUI=False, color='gray')

# Define the width and height of the bands
band_width = 50
band_height = 200

# Create high contrast band stimulus (e.g., red band)
high_contrast_band = visual.Rect(win, width=band_width, height=band_height, pos=(0, 150), fillColor='red')

# Create low contrast band stimulus (e.g., gray band)
low_contrast_band = visual.Rect(win, width=band_width, height=band_height, pos=(0, -150), fillColor='gray', lineColor='black')


# Create the arrow cue stimulus (pointing towards the top band initially)
arrow_cue = visual.TextStim(win, text='↑', pos=(0, 0), color='black', height=30)


# A variable to track which band is currently selected
selected_band = None

while True:
    # Check for user input
    for key in event.getKeys():
        if key in ['up', 'down']:
            # User selects the top band
            if key == 'up':
                selected_band = 'top'
                arrow_cue.text = '↑'
            # User selects the bottom band
            elif key == 'down':
                selected_band = 'bottom'
                arrow_cue.text = '↓'

    # Move the selected band
    if selected_band == 'top':
        if random.random() < move_right_prob_high_contrast:
            high_contrast_band.pos[1] += 2  # Move high contrast band up
        else:
            high_contrast_band.pos[1] -= 2  # Move high contrast band down
    elif selected_band == 'bottom':
        if random.random() < move_right_prob_low_contrast:
            low_contrast_band.pos[1] += 2  # Move low contrast band up
        else:
            low_contrast_band.pos[1] -= 2  # Move low contrast band down

    # Draw everything
    high_contrast_band.draw()
    low_contrast_band.draw()
    arrow_cue.draw()
    win.flip()

    # Check for quit key (escape key)
    if 'escape' in event.getKeys():
        break

# Close the window
win.close()
core.quit()
