#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on January 03, 2024, at 16:18
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'event_span'  # from the Builder filename that created this script
expInfo = {
    'session': '001',
    'participant': f"{randint(0, 999999):06.0f}",
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Namoo\\Documents\\PsychoPy\\Perception Experiment\\perception_exp_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1280, 720], fullscr=False, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome_Screen1" ---
    welcome_text1 = visual.TextStim(win=win, name='welcome_text1',
        text='Welcome to the Experiment!\n',
        font='Open Sans',
        pos=(0, .3), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcome_key1 = keyboard.Keyboard()
    press_space_text = visual.TextStim(win=win, name='press_space_text',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    intro_text = visual.TextStim(win=win, name='intro_text',
        text='In this study, we aim to understand how people perceive and recall specific day-hour combinations in two distinct calendar styles.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "Welcome_Screen2" ---
    text_desc = visual.TextStim(win=win, name='text_desc',
        text='You will be presented with cells highlighting specific day-hour combinations.\nYour task will involve recalling the highlighted information by typing the day and time. The experiment is devided into two segments. Each segment contains two parts.',
        font='Open Sans',
        pos=(0, .18), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_10 = keyboard.Keyboard()
    press_space_text_2 = visual.TextStim(win=win, name='press_space_text_2',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    tasks_text = visual.TextStim(win=win, name='tasks_text',
        text='Task Details:',
        font='Open Sans',
        pos=(0, .45), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    why_participate_text = visual.TextStim(win=win, name='why_participate_text',
        text='Why Participate?\n',
        font='Open Sans',
        pos=(0, -.15), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    participate_desc = visual.TextStim(win=win, name='participate_desc',
        text='Your participation will contribute valuable insights to our research on calendar perception.',
        font='Open Sans',
        pos=(0, -.25), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "Welcome_Screen3" ---
    privacy_text = visual.TextStim(win=win, name='privacy_text',
        text='Your Privacy Matters:\nYour responses will remain anonymous.\n\nYour Rights:\nYou have the right to stop participating at any time without consequences.\n\nConsent:\nBy completing the questionnaire, you provide consent to participate.\nThank you for contributing to our research on calendar perception! If you have any questions, please contact us at juhanjoh@ut.ee, luka.namoradze@ut.ee',
        font='Open Sans',
        pos=(0, .1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_11 = keyboard.Keyboard()
    press_space_text_3 = visual.TextStim(win=win, name='press_space_text_3',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Gender" ---
    gender_title = visual.TextStim(win=win, name='gender_title',
        text='Gender',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_gender = visual.TextStim(win=win, name='text_gender',
        text='Press:\nM - for male\nF - for female\nO - for other or I leave unanswered',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_gender = keyboard.Keyboard()
    
    # --- Initialize components for Routine "blank500" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Age" ---
    age_title = visual.TextStim(win=win, name='age_title',
        text='Age:',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    age_input = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='age_input',
         depth=-1, autoLog=True,
    )
    key_resp_12 = keyboard.Keyboard()
    endButton_2 = visual.TextStim(win=win, name='endButton_2',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "Preparation_Phase1_3" ---
    text = visual.TextStim(win=win, name='text',
        text='Press SPACE to display week layout. Familiarize yourself with it. Whenever you are ready press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Preperation_Phase2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='week7/week7_empty.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    preparation_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase3" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Press SPACE to start the first segment of the experiment. Memorize the precise day of the week and time for the highlighted area on the week layout. Press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase4" ---
    note_text = visual.TextStim(win=win, name='note_text',
        text='Note:',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    note_desc = visual.TextStim(win=win, name='note_desc',
        text='After you view highlighted cell and make sure you memorised day and time press SPACE as quickly as possible',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_13 = keyboard.Keyboard()
    press_space_text_4 = visual.TextStim(win=win, name='press_space_text_4',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "random_position" ---
    # Run 'Begin Experiment' code from rand_pos
    pos = (0, 0)
    
    # --- Initialize components for Routine "newRoutine" ---
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_week7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    recallText = visual.TextStim(win=win, name='recallText',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    endButton = visual.TextStim(win=win, name='endButton',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split1" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='The first part of the first segment is over. Press SPACE to start the second part of the segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_15 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "random_position" ---
    # Run 'Begin Experiment' code from rand_pos
    pos = (0, 0)
    
    # --- Initialize components for Routine "newRoutine" ---
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_week7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    recallText = visual.TextStim(win=win, name='recallText',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    endButton = visual.TextStim(win=win, name='endButton',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split2" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='The first segement is over. Press SPACE to start the second segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_16 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase2" ---
    preparation_text_2 = visual.TextStim(win=win, name='preparation_text_2',
        text='Press SPACE to display week layout. Familiarize yourself with it. Whenever you are ready press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase2_2" ---
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', 
        image='weekD/week-D_empty.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase3_2" ---
    memorization_text = visual.TextStim(win=win, name='memorization_text',
        text='Press SPACE to start the second segment of the experiment. Memorize the precise day of the week and time for the highlighted area on the week layout. Press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase4" ---
    note_text = visual.TextStim(win=win, name='note_text',
        text='Note:',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    note_desc = visual.TextStim(win=win, name='note_desc',
        text='After you view highlighted cell and make sure you memorised day and time press SPACE as quickly as possible',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_13 = keyboard.Keyboard()
    press_space_text_4 = visual.TextStim(win=win, name='press_space_text_4',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "random_position" ---
    # Run 'Begin Experiment' code from rand_pos
    pos = (0, 0)
    
    # --- Initialize components for Routine "image_display2" ---
    image_weekD = visual.ImageStim(
        win=win,
        name='image_weekD', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_weekD = keyboard.Keyboard()
    
    # --- Initialize components for Routine "recall2" ---
    recall_text_2 = visual.TextStim(win=win, name='recall_text_2',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    input_2 = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='input_2',
         depth=-1, autoLog=True,
    )
    end_button_2 = visual.TextStim(win=win, name='end_button_2',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_9 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation_2" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split3" ---
    split3_text = visual.TextStim(win=win, name='split3_text',
        text='The first part of the second segment is over. Press SPACE to start the second part of the segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_17 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "random_position" ---
    # Run 'Begin Experiment' code from rand_pos
    pos = (0, 0)
    
    # --- Initialize components for Routine "image_display2" ---
    image_weekD = visual.ImageStim(
        win=win,
        name='image_weekD', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_weekD = keyboard.Keyboard()
    
    # --- Initialize components for Routine "recall2" ---
    recall_text_2 = visual.TextStim(win=win, name='recall_text_2',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    input_2 = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='input_2',
         depth=-1, autoLog=True,
    )
    end_button_2 = visual.TextStim(win=win, name='end_button_2',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_9 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation_2" ---
    
    # --- Initialize components for Routine "preparation_phase2" ---
    preparation_text_2 = visual.TextStim(win=win, name='preparation_text_2',
        text='Press SPACE to display week layout. Familiarize yourself with it. Whenever you are ready press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase2_2" ---
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', 
        image='weekD/week-D_empty.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_4 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase3" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Press SPACE to start the first segment of the experiment. Memorize the precise day of the week and time for the highlighted area on the week layout. Press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase4" ---
    note_text = visual.TextStim(win=win, name='note_text',
        text='Note:',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    note_desc = visual.TextStim(win=win, name='note_desc',
        text='After you view highlighted cell and make sure you memorised day and time press SPACE as quickly as possible',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_13 = keyboard.Keyboard()
    press_space_text_4 = visual.TextStim(win=win, name='press_space_text_4',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "image_display2" ---
    image_weekD = visual.ImageStim(
        win=win,
        name='image_weekD', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_weekD = keyboard.Keyboard()
    
    # --- Initialize components for Routine "recall2" ---
    recall_text_2 = visual.TextStim(win=win, name='recall_text_2',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    input_2 = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='input_2',
         depth=-1, autoLog=True,
    )
    end_button_2 = visual.TextStim(win=win, name='end_button_2',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_9 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation_2" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split1" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='The first part of the first segment is over. Press SPACE to start the second part of the segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_15 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "image_display2" ---
    image_weekD = visual.ImageStim(
        win=win,
        name='image_weekD', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_weekD = keyboard.Keyboard()
    
    # --- Initialize components for Routine "recall2" ---
    recall_text_2 = visual.TextStim(win=win, name='recall_text_2',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    input_2 = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='input_2',
         depth=-1, autoLog=True,
    )
    end_button_2 = visual.TextStim(win=win, name='end_button_2',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_9 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation_2" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split2" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='The first segement is over. Press SPACE to start the second segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_16 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Preparation_Phase1_3" ---
    text = visual.TextStim(win=win, name='text',
        text='Press SPACE to display week layout. Familiarize yourself with it. Whenever you are ready press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard()
    
    # --- Initialize components for Routine "Preperation_Phase2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='week7/week7_empty.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    preparation_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase3_2" ---
    memorization_text = visual.TextStim(win=win, name='memorization_text',
        text='Press SPACE to start the second segment of the experiment. Memorize the precise day of the week and time for the highlighted area on the week layout. Press SPACE again to continue.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "preparation_phase4" ---
    note_text = visual.TextStim(win=win, name='note_text',
        text='Note:',
        font='Open Sans',
        pos=(0, .4), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    note_desc = visual.TextStim(win=win, name='note_desc',
        text='After you view highlighted cell and make sure you memorised day and time press SPACE as quickly as possible',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_13 = keyboard.Keyboard()
    press_space_text_4 = visual.TextStim(win=win, name='press_space_text_4',
        text='(Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "newRoutine" ---
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_week7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    recallText = visual.TextStim(win=win, name='recallText',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    endButton = visual.TextStim(win=win, name='endButton',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation" ---
    
    # --- Initialize components for Routine "break_2" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='30 second break',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_14 = keyboard.Keyboard()
    skip_break = visual.TextStim(win=win, name='skip_break',
        text='(If You Wish to Skip Break Press SPACE to continue)',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "split3" ---
    split3_text = visual.TextStim(win=win, name='split3_text',
        text='The first part of the second segment is over. Press SPACE to start the second part of the segment.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_17 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "newRoutine" ---
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(1.5, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    response_key_week7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trial" ---
    recallText = visual.TextStim(win=win, name='recallText',
        text='Recall the date and time by typing first three letters of the week followed by time (e.g. mon 8, tue 11, wed 9, thu 19, fri 15, sat 18, sun 14):',
        font='Open Sans',
        pos=(0, .35), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder=None, font='Arial',
         pos=(0, 0),     letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-1, autoLog=True,
    )
    endButton = visual.TextStim(win=win, name='endButton',
        text='Press ENTER to submit',
        font='Open Sans',
        pos=(0, -.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    key_resp_8 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "validation" ---
    
    # --- Initialize components for Routine "End_Screen" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='Congratulations, you have successfully completed all phases of the experiment! Thank you for participation.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "Welcome_Screen1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome_Screen1.started', globalClock.getTime())
    welcome_key1.keys = []
    welcome_key1.rt = []
    _welcome_key1_allKeys = []
    # Run 'Begin Routine' code from code_7
    oddParticipant = 0
    evenParticipant = 0
    session_number = int(expInfo['session'])
    if session_number % 2 == 1:
        oddParticipant = 1
    else:
        evenParticipant = 1
        
    print(session_number)
    # keep track of which components have finished
    Welcome_Screen1Components = [welcome_text1, welcome_key1, press_space_text, intro_text]
    for thisComponent in Welcome_Screen1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome_Screen1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text1* updates
        
        # if welcome_text1 is starting this frame...
        if welcome_text1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text1.frameNStart = frameN  # exact frame index
            welcome_text1.tStart = t  # local t and not account for scr refresh
            welcome_text1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text1.started')
            # update status
            welcome_text1.status = STARTED
            welcome_text1.setAutoDraw(True)
        
        # if welcome_text1 is active this frame...
        if welcome_text1.status == STARTED:
            # update params
            pass
        
        # *welcome_key1* updates
        waitOnFlip = False
        
        # if welcome_key1 is starting this frame...
        if welcome_key1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_key1.frameNStart = frameN  # exact frame index
            welcome_key1.tStart = t  # local t and not account for scr refresh
            welcome_key1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_key1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_key1.started')
            # update status
            welcome_key1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_key1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_key1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_key1.status == STARTED and not waitOnFlip:
            theseKeys = welcome_key1.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_key1_allKeys.extend(theseKeys)
            if len(_welcome_key1_allKeys):
                welcome_key1.keys = _welcome_key1_allKeys[-1].name  # just the last key pressed
                welcome_key1.rt = _welcome_key1_allKeys[-1].rt
                welcome_key1.duration = _welcome_key1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *press_space_text* updates
        
        # if press_space_text is starting this frame...
        if press_space_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_text.frameNStart = frameN  # exact frame index
            press_space_text.tStart = t  # local t and not account for scr refresh
            press_space_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'press_space_text.started')
            # update status
            press_space_text.status = STARTED
            press_space_text.setAutoDraw(True)
        
        # if press_space_text is active this frame...
        if press_space_text.status == STARTED:
            # update params
            pass
        
        # *intro_text* updates
        
        # if intro_text is starting this frame...
        if intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            intro_text.frameNStart = frameN  # exact frame index
            intro_text.tStart = t  # local t and not account for scr refresh
            intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'intro_text.started')
            # update status
            intro_text.status = STARTED
            intro_text.setAutoDraw(True)
        
        # if intro_text is active this frame...
        if intro_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome_Screen1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome_Screen1" ---
    for thisComponent in Welcome_Screen1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome_Screen1.stopped', globalClock.getTime())
    # check responses
    if welcome_key1.keys in ['', [], None]:  # No response was made
        welcome_key1.keys = None
    thisExp.addData('welcome_key1.keys',welcome_key1.keys)
    if welcome_key1.keys != None:  # we had a response
        thisExp.addData('welcome_key1.rt', welcome_key1.rt)
        thisExp.addData('welcome_key1.duration', welcome_key1.duration)
    thisExp.nextEntry()
    # the Routine "Welcome_Screen1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Welcome_Screen2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome_Screen2.started', globalClock.getTime())
    key_resp_10.keys = []
    key_resp_10.rt = []
    _key_resp_10_allKeys = []
    # keep track of which components have finished
    Welcome_Screen2Components = [text_desc, key_resp_10, press_space_text_2, tasks_text, why_participate_text, participate_desc]
    for thisComponent in Welcome_Screen2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome_Screen2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_desc* updates
        
        # if text_desc is starting this frame...
        if text_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_desc.frameNStart = frameN  # exact frame index
            text_desc.tStart = t  # local t and not account for scr refresh
            text_desc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_desc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_desc.started')
            # update status
            text_desc.status = STARTED
            text_desc.setAutoDraw(True)
        
        # if text_desc is active this frame...
        if text_desc.status == STARTED:
            # update params
            pass
        
        # *key_resp_10* updates
        waitOnFlip = False
        
        # if key_resp_10 is starting this frame...
        if key_resp_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_10.frameNStart = frameN  # exact frame index
            key_resp_10.tStart = t  # local t and not account for scr refresh
            key_resp_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_10.started')
            # update status
            key_resp_10.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_10.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_10.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_10.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_10.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_10_allKeys.extend(theseKeys)
            if len(_key_resp_10_allKeys):
                key_resp_10.keys = _key_resp_10_allKeys[-1].name  # just the last key pressed
                key_resp_10.rt = _key_resp_10_allKeys[-1].rt
                key_resp_10.duration = _key_resp_10_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *press_space_text_2* updates
        
        # if press_space_text_2 is starting this frame...
        if press_space_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_text_2.frameNStart = frameN  # exact frame index
            press_space_text_2.tStart = t  # local t and not account for scr refresh
            press_space_text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'press_space_text_2.started')
            # update status
            press_space_text_2.status = STARTED
            press_space_text_2.setAutoDraw(True)
        
        # if press_space_text_2 is active this frame...
        if press_space_text_2.status == STARTED:
            # update params
            pass
        
        # *tasks_text* updates
        
        # if tasks_text is starting this frame...
        if tasks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            tasks_text.frameNStart = frameN  # exact frame index
            tasks_text.tStart = t  # local t and not account for scr refresh
            tasks_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(tasks_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'tasks_text.started')
            # update status
            tasks_text.status = STARTED
            tasks_text.setAutoDraw(True)
        
        # if tasks_text is active this frame...
        if tasks_text.status == STARTED:
            # update params
            pass
        
        # *why_participate_text* updates
        
        # if why_participate_text is starting this frame...
        if why_participate_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            why_participate_text.frameNStart = frameN  # exact frame index
            why_participate_text.tStart = t  # local t and not account for scr refresh
            why_participate_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(why_participate_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'why_participate_text.started')
            # update status
            why_participate_text.status = STARTED
            why_participate_text.setAutoDraw(True)
        
        # if why_participate_text is active this frame...
        if why_participate_text.status == STARTED:
            # update params
            pass
        
        # *participate_desc* updates
        
        # if participate_desc is starting this frame...
        if participate_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            participate_desc.frameNStart = frameN  # exact frame index
            participate_desc.tStart = t  # local t and not account for scr refresh
            participate_desc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(participate_desc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'participate_desc.started')
            # update status
            participate_desc.status = STARTED
            participate_desc.setAutoDraw(True)
        
        # if participate_desc is active this frame...
        if participate_desc.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome_Screen2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome_Screen2" ---
    for thisComponent in Welcome_Screen2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome_Screen2.stopped', globalClock.getTime())
    # check responses
    if key_resp_10.keys in ['', [], None]:  # No response was made
        key_resp_10.keys = None
    thisExp.addData('key_resp_10.keys',key_resp_10.keys)
    if key_resp_10.keys != None:  # we had a response
        thisExp.addData('key_resp_10.rt', key_resp_10.rt)
        thisExp.addData('key_resp_10.duration', key_resp_10.duration)
    thisExp.nextEntry()
    # the Routine "Welcome_Screen2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Welcome_Screen3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Welcome_Screen3.started', globalClock.getTime())
    key_resp_11.keys = []
    key_resp_11.rt = []
    _key_resp_11_allKeys = []
    # keep track of which components have finished
    Welcome_Screen3Components = [privacy_text, key_resp_11, press_space_text_3]
    for thisComponent in Welcome_Screen3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome_Screen3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *privacy_text* updates
        
        # if privacy_text is starting this frame...
        if privacy_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            privacy_text.frameNStart = frameN  # exact frame index
            privacy_text.tStart = t  # local t and not account for scr refresh
            privacy_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(privacy_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'privacy_text.started')
            # update status
            privacy_text.status = STARTED
            privacy_text.setAutoDraw(True)
        
        # if privacy_text is active this frame...
        if privacy_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_11* updates
        waitOnFlip = False
        
        # if key_resp_11 is starting this frame...
        if key_resp_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_11.frameNStart = frameN  # exact frame index
            key_resp_11.tStart = t  # local t and not account for scr refresh
            key_resp_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_11.started')
            # update status
            key_resp_11.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_11.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_11.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_11.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_11.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_11_allKeys.extend(theseKeys)
            if len(_key_resp_11_allKeys):
                key_resp_11.keys = _key_resp_11_allKeys[-1].name  # just the last key pressed
                key_resp_11.rt = _key_resp_11_allKeys[-1].rt
                key_resp_11.duration = _key_resp_11_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *press_space_text_3* updates
        
        # if press_space_text_3 is starting this frame...
        if press_space_text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            press_space_text_3.frameNStart = frameN  # exact frame index
            press_space_text_3.tStart = t  # local t and not account for scr refresh
            press_space_text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(press_space_text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'press_space_text_3.started')
            # update status
            press_space_text_3.status = STARTED
            press_space_text_3.setAutoDraw(True)
        
        # if press_space_text_3 is active this frame...
        if press_space_text_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome_Screen3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome_Screen3" ---
    for thisComponent in Welcome_Screen3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Welcome_Screen3.stopped', globalClock.getTime())
    # check responses
    if key_resp_11.keys in ['', [], None]:  # No response was made
        key_resp_11.keys = None
    thisExp.addData('key_resp_11.keys',key_resp_11.keys)
    if key_resp_11.keys != None:  # we had a response
        thisExp.addData('key_resp_11.rt', key_resp_11.rt)
        thisExp.addData('key_resp_11.duration', key_resp_11.duration)
    thisExp.nextEntry()
    # the Routine "Welcome_Screen3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Gender" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Gender.started', globalClock.getTime())
    key_gender.keys = []
    key_gender.rt = []
    _key_gender_allKeys = []
    # keep track of which components have finished
    GenderComponents = [gender_title, text_gender, key_gender]
    for thisComponent in GenderComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Gender" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *gender_title* updates
        
        # if gender_title is starting this frame...
        if gender_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            gender_title.frameNStart = frameN  # exact frame index
            gender_title.tStart = t  # local t and not account for scr refresh
            gender_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gender_title, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'gender_title.started')
            # update status
            gender_title.status = STARTED
            gender_title.setAutoDraw(True)
        
        # if gender_title is active this frame...
        if gender_title.status == STARTED:
            # update params
            pass
        
        # *text_gender* updates
        
        # if text_gender is starting this frame...
        if text_gender.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_gender.frameNStart = frameN  # exact frame index
            text_gender.tStart = t  # local t and not account for scr refresh
            text_gender.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_gender, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_gender.started')
            # update status
            text_gender.status = STARTED
            text_gender.setAutoDraw(True)
        
        # if text_gender is active this frame...
        if text_gender.status == STARTED:
            # update params
            pass
        
        # *key_gender* updates
        waitOnFlip = False
        
        # if key_gender is starting this frame...
        if key_gender.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_gender.frameNStart = frameN  # exact frame index
            key_gender.tStart = t  # local t and not account for scr refresh
            key_gender.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_gender, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_gender.started')
            # update status
            key_gender.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_gender.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_gender.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_gender.status == STARTED and not waitOnFlip:
            theseKeys = key_gender.getKeys(keyList=['m', 'f', 'o'], ignoreKeys=["escape"], waitRelease=False)
            _key_gender_allKeys.extend(theseKeys)
            if len(_key_gender_allKeys):
                key_gender.keys = _key_gender_allKeys[-1].name  # just the last key pressed
                key_gender.rt = _key_gender_allKeys[-1].rt
                key_gender.duration = _key_gender_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in GenderComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Gender" ---
    for thisComponent in GenderComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Gender.stopped', globalClock.getTime())
    # check responses
    if key_gender.keys in ['', [], None]:  # No response was made
        key_gender.keys = None
    thisExp.addData('key_gender.keys',key_gender.keys)
    if key_gender.keys != None:  # we had a response
        thisExp.addData('key_gender.rt', key_gender.rt)
        thisExp.addData('key_gender.duration', key_gender.duration)
    thisExp.nextEntry()
    # the Routine "Gender" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "blank500" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('blank500.started', globalClock.getTime())
    # keep track of which components have finished
    blank500Components = [text_5]
    for thisComponent in blank500Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank500" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # if text_5 is stopping this frame...
        if text_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_5.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.stopped')
                # update status
                text_5.status = FINISHED
                text_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank500Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank500" ---
    for thisComponent in blank500Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('blank500.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.500000)
    
    # --- Prepare to start Routine "Age" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Age.started', globalClock.getTime())
    age_input.reset()
    age_input.setText('')
    key_resp_12.keys = []
    key_resp_12.rt = []
    _key_resp_12_allKeys = []
    # keep track of which components have finished
    AgeComponents = [age_title, age_input, key_resp_12, endButton_2]
    for thisComponent in AgeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Age" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *age_title* updates
        
        # if age_title is starting this frame...
        if age_title.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            age_title.frameNStart = frameN  # exact frame index
            age_title.tStart = t  # local t and not account for scr refresh
            age_title.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(age_title, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'age_title.started')
            # update status
            age_title.status = STARTED
            age_title.setAutoDraw(True)
        
        # if age_title is active this frame...
        if age_title.status == STARTED:
            # update params
            pass
        
        # *age_input* updates
        
        # if age_input is starting this frame...
        if age_input.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            age_input.frameNStart = frameN  # exact frame index
            age_input.tStart = t  # local t and not account for scr refresh
            age_input.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(age_input, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'age_input.started')
            # update status
            age_input.status = STARTED
            age_input.setAutoDraw(True)
        
        # if age_input is active this frame...
        if age_input.status == STARTED:
            # update params
            pass
        
        # *key_resp_12* updates
        waitOnFlip = False
        
        # if key_resp_12 is starting this frame...
        if key_resp_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_12.frameNStart = frameN  # exact frame index
            key_resp_12.tStart = t  # local t and not account for scr refresh
            key_resp_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_12.started')
            # update status
            key_resp_12.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_12.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_12.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_12.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_12.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_12_allKeys.extend(theseKeys)
            if len(_key_resp_12_allKeys):
                key_resp_12.keys = _key_resp_12_allKeys[-1].name  # just the last key pressed
                key_resp_12.rt = _key_resp_12_allKeys[-1].rt
                key_resp_12.duration = _key_resp_12_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *endButton_2* updates
        
        # if endButton_2 is starting this frame...
        if endButton_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            endButton_2.frameNStart = frameN  # exact frame index
            endButton_2.tStart = t  # local t and not account for scr refresh
            endButton_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(endButton_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'endButton_2.started')
            # update status
            endButton_2.status = STARTED
            endButton_2.setAutoDraw(True)
        
        # if endButton_2 is active this frame...
        if endButton_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in AgeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Age" ---
    for thisComponent in AgeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Age.stopped', globalClock.getTime())
    thisExp.addData('age_input.text',age_input.text)
    # check responses
    if key_resp_12.keys in ['', [], None]:  # No response was made
        key_resp_12.keys = None
    thisExp.addData('key_resp_12.keys',key_resp_12.keys)
    if key_resp_12.keys != None:  # we had a response
        thisExp.addData('key_resp_12.rt', key_resp_12.rt)
        thisExp.addData('key_resp_12.duration', key_resp_12.duration)
    thisExp.nextEntry()
    # the Routine "Age" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_odd = data.TrialHandler(nReps=oddParticipant, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_odd')
    thisExp.addLoop(trials_odd)  # add the loop to the experiment
    thisTrials_odd = trials_odd.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_odd.rgb)
    if thisTrials_odd != None:
        for paramName in thisTrials_odd:
            globals()[paramName] = thisTrials_odd[paramName]
    
    for thisTrials_odd in trials_odd:
        currentLoop = trials_odd
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_odd.rgb)
        if thisTrials_odd != None:
            for paramName in thisTrials_odd:
                globals()[paramName] = thisTrials_odd[paramName]
        
        # --- Prepare to start Routine "Preparation_Phase1_3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Preparation_Phase1_3.started', globalClock.getTime())
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        Preparation_Phase1_3Components = [text, key_resp]
        for thisComponent in Preparation_Phase1_3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Preparation_Phase1_3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preparation_Phase1_3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preparation_Phase1_3" ---
        for thisComponent in Preparation_Phase1_3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Preparation_Phase1_3.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_odd.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_odd.addData('key_resp.rt', key_resp.rt)
            trials_odd.addData('key_resp.duration', key_resp.duration)
        # the Routine "Preparation_Phase1_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Preperation_Phase2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Preperation_Phase2.started', globalClock.getTime())
        preparation_key.keys = []
        preparation_key.rt = []
        _preparation_key_allKeys = []
        # keep track of which components have finished
        Preperation_Phase2Components = [image_3, preparation_key]
        for thisComponent in Preperation_Phase2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Preperation_Phase2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_3* updates
            
            # if image_3 is starting this frame...
            if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_3.frameNStart = frameN  # exact frame index
                image_3.tStart = t  # local t and not account for scr refresh
                image_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_3.started')
                # update status
                image_3.status = STARTED
                image_3.setAutoDraw(True)
            
            # if image_3 is active this frame...
            if image_3.status == STARTED:
                # update params
                pass
            
            # if image_3 is stopping this frame...
            if image_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_3.tStartRefresh + 120-frameTolerance:
                    # keep track of stop time/frame for later
                    image_3.tStop = t  # not accounting for scr refresh
                    image_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.stopped')
                    # update status
                    image_3.status = FINISHED
                    image_3.setAutoDraw(False)
            
            # *preparation_key* updates
            waitOnFlip = False
            
            # if preparation_key is starting this frame...
            if preparation_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preparation_key.frameNStart = frameN  # exact frame index
                preparation_key.tStart = t  # local t and not account for scr refresh
                preparation_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preparation_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'preparation_key.started')
                # update status
                preparation_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(preparation_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(preparation_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if preparation_key.status == STARTED and not waitOnFlip:
                theseKeys = preparation_key.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _preparation_key_allKeys.extend(theseKeys)
                if len(_preparation_key_allKeys):
                    preparation_key.keys = _preparation_key_allKeys[-1].name  # just the last key pressed
                    preparation_key.rt = _preparation_key_allKeys[-1].rt
                    preparation_key.duration = _preparation_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preperation_Phase2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preperation_Phase2" ---
        for thisComponent in Preperation_Phase2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Preperation_Phase2.stopped', globalClock.getTime())
        # check responses
        if preparation_key.keys in ['', [], None]:  # No response was made
            preparation_key.keys = None
        trials_odd.addData('preparation_key.keys',preparation_key.keys)
        if preparation_key.keys != None:  # we had a response
            trials_odd.addData('preparation_key.rt', preparation_key.rt)
            trials_odd.addData('preparation_key.duration', preparation_key.duration)
        # the Routine "Preperation_Phase2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase3.started', globalClock.getTime())
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        preparation_phase3Components = [text_2, key_resp_2]
        for thisComponent in preparation_phase3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase3" ---
        for thisComponent in preparation_phase3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase3.stopped', globalClock.getTime())
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        trials_odd.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            trials_odd.addData('key_resp_2.rt', key_resp_2.rt)
            trials_odd.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "preparation_phase3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase4.started', globalClock.getTime())
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # keep track of which components have finished
        preparation_phase4Components = [note_text, note_desc, key_resp_13, press_space_text_4]
        for thisComponent in preparation_phase4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *note_text* updates
            
            # if note_text is starting this frame...
            if note_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_text.frameNStart = frameN  # exact frame index
                note_text.tStart = t  # local t and not account for scr refresh
                note_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_text.started')
                # update status
                note_text.status = STARTED
                note_text.setAutoDraw(True)
            
            # if note_text is active this frame...
            if note_text.status == STARTED:
                # update params
                pass
            
            # *note_desc* updates
            
            # if note_desc is starting this frame...
            if note_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_desc.frameNStart = frameN  # exact frame index
                note_desc.tStart = t  # local t and not account for scr refresh
                note_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_desc.started')
                # update status
                note_desc.status = STARTED
                note_desc.setAutoDraw(True)
            
            # if note_desc is active this frame...
            if note_desc.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *press_space_text_4* updates
            
            # if press_space_text_4 is starting this frame...
            if press_space_text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                press_space_text_4.frameNStart = frameN  # exact frame index
                press_space_text_4.tStart = t  # local t and not account for scr refresh
                press_space_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_space_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_space_text_4.started')
                # update status
                press_space_text_4.status = STARTED
                press_space_text_4.setAutoDraw(True)
            
            # if press_space_text_4 is active this frame...
            if press_space_text_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase4" ---
        for thisComponent in preparation_phase4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase4.stopped', globalClock.getTime())
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials_odd.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials_odd.addData('key_resp_13.rt', key_resp_13.rt)
            trials_odd.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "preparation_phase4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_week7.xlsx'),
            seed=None, name='trials')
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "random_position" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('random_position.started', globalClock.getTime())
            # keep track of which components have finished
            random_positionComponents = []
            for thisComponent in random_positionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "random_position" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from rand_pos
                import random
                
                random_number = random.randint(0, 2)
                print("random number = ", random_number)
                
                if(random_number == 1):
                    pos = (-.15, 0)
                elif(random_number == 2):
                    pos = (.15,0)
                else:
                    pos = (0,0)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in random_positionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "random_position" ---
            for thisComponent in random_positionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('random_position.stopped', globalClock.getTime())
            # the Routine "random_position" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "newRoutine" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('newRoutine.started', globalClock.getTime())
            image_2.setPos(pos)
            image_2.setImage(imageName)
            response_key_week7.keys = []
            response_key_week7.rt = []
            _response_key_week7_allKeys = []
            # keep track of which components have finished
            newRoutineComponents = [image_2, response_key_week7]
            for thisComponent in newRoutineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "newRoutine" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # *response_key_week7* updates
                waitOnFlip = False
                
                # if response_key_week7 is starting this frame...
                if response_key_week7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_week7.frameNStart = frameN  # exact frame index
                    response_key_week7.tStart = t  # local t and not account for scr refresh
                    response_key_week7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_week7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_week7.started')
                    # update status
                    response_key_week7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_week7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_week7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_week7.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_week7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_week7_allKeys.extend(theseKeys)
                    if len(_response_key_week7_allKeys):
                        response_key_week7.keys = _response_key_week7_allKeys[-1].name  # just the last key pressed
                        response_key_week7.rt = _response_key_week7_allKeys[-1].rt
                        response_key_week7.duration = _response_key_week7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in newRoutineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "newRoutine" ---
            for thisComponent in newRoutineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('newRoutine.stopped', globalClock.getTime())
            # check responses
            if response_key_week7.keys in ['', [], None]:  # No response was made
                response_key_week7.keys = None
            trials.addData('response_key_week7.keys',response_key_week7.keys)
            if response_key_week7.keys != None:  # we had a response
                trials.addData('response_key_week7.rt', response_key_week7.rt)
                trials.addData('response_key_week7.duration', response_key_week7.duration)
            # the Routine "newRoutine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            textbox.reset()
            textbox.setText('')
            # Run 'Begin Routine' code from code_5
            feedback_message = imageName
            print(feedback_message)
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # keep track of which components have finished
            trialComponents = [recallText, textbox, endButton, key_resp_8]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recallText* updates
                
                # if recallText is starting this frame...
                if recallText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recallText.frameNStart = frameN  # exact frame index
                    recallText.tStart = t  # local t and not account for scr refresh
                    recallText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recallText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recallText.started')
                    # update status
                    recallText.status = STARTED
                    recallText.setAutoDraw(True)
                
                # if recallText is active this frame...
                if recallText.status == STARTED:
                    # update params
                    pass
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.started')
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                
                # *endButton* updates
                
                # if endButton is starting this frame...
                if endButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    endButton.frameNStart = frameN  # exact frame index
                    endButton.tStart = t  # local t and not account for scr refresh
                    endButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(endButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'endButton.started')
                    # update status
                    endButton.status = STARTED
                    endButton.setAutoDraw(True)
                
                # if endButton is active this frame...
                if endButton.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            trials.addData('textbox.text',textbox.text)
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            trials.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                trials.addData('key_resp_8.rt', key_resp_8.rt)
                trials.addData('key_resp_8.duration', key_resp_8.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_3
            response = textbox.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            
            print("correct ans", correctAns)
            print("response", response)
            print("feedback", feedbackMessage)
            # keep track of which components have finished
            validationComponents = []
            for thisComponent in validationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation" ---
            for thisComponent in validationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation.stopped', globalClock.getTime())
            # the Routine "validation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_odd.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_odd.addData('key_resp_14.rt', key_resp_14.rt)
            trials_odd.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split1.started', globalClock.getTime())
        key_resp_15.keys = []
        key_resp_15.rt = []
        _key_resp_15_allKeys = []
        # keep track of which components have finished
        split1Components = [text_6, key_resp_15]
        for thisComponent in split1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *key_resp_15* updates
            waitOnFlip = False
            
            # if key_resp_15 is starting this frame...
            if key_resp_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_15.frameNStart = frameN  # exact frame index
                key_resp_15.tStart = t  # local t and not account for scr refresh
                key_resp_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_15.started')
                # update status
                key_resp_15.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_15.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_15.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_15.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_15.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_15_allKeys.extend(theseKeys)
                if len(_key_resp_15_allKeys):
                    key_resp_15.keys = _key_resp_15_allKeys[-1].name  # just the last key pressed
                    key_resp_15.rt = _key_resp_15_allKeys[-1].rt
                    key_resp_15.duration = _key_resp_15_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split1" ---
        for thisComponent in split1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split1.stopped', globalClock.getTime())
        # check responses
        if key_resp_15.keys in ['', [], None]:  # No response was made
            key_resp_15.keys = None
        trials_odd.addData('key_resp_15.keys',key_resp_15.keys)
        if key_resp_15.keys != None:  # we had a response
            trials_odd.addData('key_resp_15.rt', key_resp_15.rt)
            trials_odd.addData('key_resp_15.duration', key_resp_15.duration)
        # the Routine "split1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials2 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_week7.xlsx'),
            seed=None, name='trials2')
        thisExp.addLoop(trials2)  # add the loop to the experiment
        thisTrials2 = trials2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials2.rgb)
        if thisTrials2 != None:
            for paramName in thisTrials2:
                globals()[paramName] = thisTrials2[paramName]
        
        for thisTrials2 in trials2:
            currentLoop = trials2
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials2.rgb)
            if thisTrials2 != None:
                for paramName in thisTrials2:
                    globals()[paramName] = thisTrials2[paramName]
            
            # --- Prepare to start Routine "random_position" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('random_position.started', globalClock.getTime())
            # keep track of which components have finished
            random_positionComponents = []
            for thisComponent in random_positionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "random_position" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from rand_pos
                import random
                
                random_number = random.randint(0, 2)
                print("random number = ", random_number)
                
                if(random_number == 1):
                    pos = (-.15, 0)
                elif(random_number == 2):
                    pos = (.15,0)
                else:
                    pos = (0,0)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in random_positionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "random_position" ---
            for thisComponent in random_positionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('random_position.stopped', globalClock.getTime())
            # the Routine "random_position" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "newRoutine" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('newRoutine.started', globalClock.getTime())
            image_2.setPos(pos)
            image_2.setImage(imageName)
            response_key_week7.keys = []
            response_key_week7.rt = []
            _response_key_week7_allKeys = []
            # keep track of which components have finished
            newRoutineComponents = [image_2, response_key_week7]
            for thisComponent in newRoutineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "newRoutine" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # *response_key_week7* updates
                waitOnFlip = False
                
                # if response_key_week7 is starting this frame...
                if response_key_week7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_week7.frameNStart = frameN  # exact frame index
                    response_key_week7.tStart = t  # local t and not account for scr refresh
                    response_key_week7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_week7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_week7.started')
                    # update status
                    response_key_week7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_week7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_week7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_week7.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_week7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_week7_allKeys.extend(theseKeys)
                    if len(_response_key_week7_allKeys):
                        response_key_week7.keys = _response_key_week7_allKeys[-1].name  # just the last key pressed
                        response_key_week7.rt = _response_key_week7_allKeys[-1].rt
                        response_key_week7.duration = _response_key_week7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in newRoutineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "newRoutine" ---
            for thisComponent in newRoutineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('newRoutine.stopped', globalClock.getTime())
            # check responses
            if response_key_week7.keys in ['', [], None]:  # No response was made
                response_key_week7.keys = None
            trials2.addData('response_key_week7.keys',response_key_week7.keys)
            if response_key_week7.keys != None:  # we had a response
                trials2.addData('response_key_week7.rt', response_key_week7.rt)
                trials2.addData('response_key_week7.duration', response_key_week7.duration)
            # the Routine "newRoutine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            textbox.reset()
            textbox.setText('')
            # Run 'Begin Routine' code from code_5
            feedback_message = imageName
            print(feedback_message)
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # keep track of which components have finished
            trialComponents = [recallText, textbox, endButton, key_resp_8]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recallText* updates
                
                # if recallText is starting this frame...
                if recallText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recallText.frameNStart = frameN  # exact frame index
                    recallText.tStart = t  # local t and not account for scr refresh
                    recallText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recallText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recallText.started')
                    # update status
                    recallText.status = STARTED
                    recallText.setAutoDraw(True)
                
                # if recallText is active this frame...
                if recallText.status == STARTED:
                    # update params
                    pass
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.started')
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                
                # *endButton* updates
                
                # if endButton is starting this frame...
                if endButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    endButton.frameNStart = frameN  # exact frame index
                    endButton.tStart = t  # local t and not account for scr refresh
                    endButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(endButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'endButton.started')
                    # update status
                    endButton.status = STARTED
                    endButton.setAutoDraw(True)
                
                # if endButton is active this frame...
                if endButton.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            trials2.addData('textbox.text',textbox.text)
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            trials2.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                trials2.addData('key_resp_8.rt', key_resp_8.rt)
                trials2.addData('key_resp_8.duration', key_resp_8.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_3
            response = textbox.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            
            print("correct ans", correctAns)
            print("response", response)
            print("feedback", feedbackMessage)
            # keep track of which components have finished
            validationComponents = []
            for thisComponent in validationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation" ---
            for thisComponent in validationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation.stopped', globalClock.getTime())
            # the Routine "validation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials2'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_odd.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_odd.addData('key_resp_14.rt', key_resp_14.rt)
            trials_odd.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split2.started', globalClock.getTime())
        key_resp_16.keys = []
        key_resp_16.rt = []
        _key_resp_16_allKeys = []
        # keep track of which components have finished
        split2Components = [text_9, key_resp_16]
        for thisComponent in split2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *key_resp_16* updates
            waitOnFlip = False
            
            # if key_resp_16 is starting this frame...
            if key_resp_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_16.frameNStart = frameN  # exact frame index
                key_resp_16.tStart = t  # local t and not account for scr refresh
                key_resp_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_16.started')
                # update status
                key_resp_16.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_16.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_16.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_16.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_16.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_16_allKeys.extend(theseKeys)
                if len(_key_resp_16_allKeys):
                    key_resp_16.keys = _key_resp_16_allKeys[-1].name  # just the last key pressed
                    key_resp_16.rt = _key_resp_16_allKeys[-1].rt
                    key_resp_16.duration = _key_resp_16_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split2" ---
        for thisComponent in split2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split2.stopped', globalClock.getTime())
        # check responses
        if key_resp_16.keys in ['', [], None]:  # No response was made
            key_resp_16.keys = None
        trials_odd.addData('key_resp_16.keys',key_resp_16.keys)
        if key_resp_16.keys != None:  # we had a response
            trials_odd.addData('key_resp_16.rt', key_resp_16.rt)
            trials_odd.addData('key_resp_16.duration', key_resp_16.duration)
        # the Routine "split2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase2.started', globalClock.getTime())
        key_resp_7.keys = []
        key_resp_7.rt = []
        _key_resp_7_allKeys = []
        # keep track of which components have finished
        preparation_phase2Components = [preparation_text_2, key_resp_7]
        for thisComponent in preparation_phase2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *preparation_text_2* updates
            
            # if preparation_text_2 is starting this frame...
            if preparation_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preparation_text_2.frameNStart = frameN  # exact frame index
                preparation_text_2.tStart = t  # local t and not account for scr refresh
                preparation_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preparation_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'preparation_text_2.started')
                # update status
                preparation_text_2.status = STARTED
                preparation_text_2.setAutoDraw(True)
            
            # if preparation_text_2 is active this frame...
            if preparation_text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_7* updates
            waitOnFlip = False
            
            # if key_resp_7 is starting this frame...
            if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_7.frameNStart = frameN  # exact frame index
                key_resp_7.tStart = t  # local t and not account for scr refresh
                key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_7.started')
                # update status
                key_resp_7.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_7.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_7_allKeys.extend(theseKeys)
                if len(_key_resp_7_allKeys):
                    key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                    key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                    key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase2" ---
        for thisComponent in preparation_phase2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase2.stopped', globalClock.getTime())
        # check responses
        if key_resp_7.keys in ['', [], None]:  # No response was made
            key_resp_7.keys = None
        trials_odd.addData('key_resp_7.keys',key_resp_7.keys)
        if key_resp_7.keys != None:  # we had a response
            trials_odd.addData('key_resp_7.rt', key_resp_7.rt)
            trials_odd.addData('key_resp_7.duration', key_resp_7.duration)
        # the Routine "preparation_phase2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase2_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase2_2.started', globalClock.getTime())
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        preparation_phase2_2Components = [image_4, key_resp_4]
        for thisComponent in preparation_phase2_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase2_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_4* updates
            
            # if image_4 is starting this frame...
            if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_4.frameNStart = frameN  # exact frame index
                image_4.tStart = t  # local t and not account for scr refresh
                image_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_4.started')
                # update status
                image_4.status = STARTED
                image_4.setAutoDraw(True)
            
            # if image_4 is active this frame...
            if image_4.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase2_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase2_2" ---
        for thisComponent in preparation_phase2_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase2_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        trials_odd.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            trials_odd.addData('key_resp_4.rt', key_resp_4.rt)
            trials_odd.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "preparation_phase2_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase3_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase3_2.started', globalClock.getTime())
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # keep track of which components have finished
        preparation_phase3_2Components = [memorization_text, key_resp_5]
        for thisComponent in preparation_phase3_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase3_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *memorization_text* updates
            
            # if memorization_text is starting this frame...
            if memorization_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                memorization_text.frameNStart = frameN  # exact frame index
                memorization_text.tStart = t  # local t and not account for scr refresh
                memorization_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memorization_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'memorization_text.started')
                # update status
                memorization_text.status = STARTED
                memorization_text.setAutoDraw(True)
            
            # if memorization_text is active this frame...
            if memorization_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                    key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                    key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase3_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase3_2" ---
        for thisComponent in preparation_phase3_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase3_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        trials_odd.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            trials_odd.addData('key_resp_5.rt', key_resp_5.rt)
            trials_odd.addData('key_resp_5.duration', key_resp_5.duration)
        # the Routine "preparation_phase3_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase4.started', globalClock.getTime())
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # keep track of which components have finished
        preparation_phase4Components = [note_text, note_desc, key_resp_13, press_space_text_4]
        for thisComponent in preparation_phase4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *note_text* updates
            
            # if note_text is starting this frame...
            if note_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_text.frameNStart = frameN  # exact frame index
                note_text.tStart = t  # local t and not account for scr refresh
                note_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_text.started')
                # update status
                note_text.status = STARTED
                note_text.setAutoDraw(True)
            
            # if note_text is active this frame...
            if note_text.status == STARTED:
                # update params
                pass
            
            # *note_desc* updates
            
            # if note_desc is starting this frame...
            if note_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_desc.frameNStart = frameN  # exact frame index
                note_desc.tStart = t  # local t and not account for scr refresh
                note_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_desc.started')
                # update status
                note_desc.status = STARTED
                note_desc.setAutoDraw(True)
            
            # if note_desc is active this frame...
            if note_desc.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *press_space_text_4* updates
            
            # if press_space_text_4 is starting this frame...
            if press_space_text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                press_space_text_4.frameNStart = frameN  # exact frame index
                press_space_text_4.tStart = t  # local t and not account for scr refresh
                press_space_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_space_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_space_text_4.started')
                # update status
                press_space_text_4.status = STARTED
                press_space_text_4.setAutoDraw(True)
            
            # if press_space_text_4 is active this frame...
            if press_space_text_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase4" ---
        for thisComponent in preparation_phase4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase4.stopped', globalClock.getTime())
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials_odd.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials_odd.addData('key_resp_13.rt', key_resp_13.rt)
            trials_odd.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "preparation_phase4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_2 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_weekD.xlsx'),
            seed=None, name='trials_2')
        thisExp.addLoop(trials_2)  # add the loop to the experiment
        thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        for thisTrial_2 in trials_2:
            currentLoop = trials_2
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
            if thisTrial_2 != None:
                for paramName in thisTrial_2:
                    globals()[paramName] = thisTrial_2[paramName]
            
            # --- Prepare to start Routine "random_position" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('random_position.started', globalClock.getTime())
            # keep track of which components have finished
            random_positionComponents = []
            for thisComponent in random_positionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "random_position" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from rand_pos
                import random
                
                random_number = random.randint(0, 2)
                print("random number = ", random_number)
                
                if(random_number == 1):
                    pos = (-.15, 0)
                elif(random_number == 2):
                    pos = (.15,0)
                else:
                    pos = (0,0)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in random_positionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "random_position" ---
            for thisComponent in random_positionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('random_position.stopped', globalClock.getTime())
            # the Routine "random_position" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "image_display2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('image_display2.started', globalClock.getTime())
            image_weekD.setPos(pos)
            image_weekD.setImage(imageName)
            response_key_weekD.keys = []
            response_key_weekD.rt = []
            _response_key_weekD_allKeys = []
            # keep track of which components have finished
            image_display2Components = [image_weekD, response_key_weekD]
            for thisComponent in image_display2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "image_display2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_weekD* updates
                
                # if image_weekD is starting this frame...
                if image_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_weekD.frameNStart = frameN  # exact frame index
                    image_weekD.tStart = t  # local t and not account for scr refresh
                    image_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_weekD.started')
                    # update status
                    image_weekD.status = STARTED
                    image_weekD.setAutoDraw(True)
                
                # if image_weekD is active this frame...
                if image_weekD.status == STARTED:
                    # update params
                    pass
                
                # *response_key_weekD* updates
                waitOnFlip = False
                
                # if response_key_weekD is starting this frame...
                if response_key_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_weekD.frameNStart = frameN  # exact frame index
                    response_key_weekD.tStart = t  # local t and not account for scr refresh
                    response_key_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_weekD.started')
                    # update status
                    response_key_weekD.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_weekD.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_weekD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_weekD.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_weekD.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_weekD_allKeys.extend(theseKeys)
                    if len(_response_key_weekD_allKeys):
                        response_key_weekD.keys = _response_key_weekD_allKeys[-1].name  # just the last key pressed
                        response_key_weekD.rt = _response_key_weekD_allKeys[-1].rt
                        response_key_weekD.duration = _response_key_weekD_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_display2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_display2" ---
            for thisComponent in image_display2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('image_display2.stopped', globalClock.getTime())
            # check responses
            if response_key_weekD.keys in ['', [], None]:  # No response was made
                response_key_weekD.keys = None
            trials_2.addData('response_key_weekD.keys',response_key_weekD.keys)
            if response_key_weekD.keys != None:  # we had a response
                trials_2.addData('response_key_weekD.rt', response_key_weekD.rt)
                trials_2.addData('response_key_weekD.duration', response_key_weekD.duration)
            # the Routine "image_display2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "recall2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('recall2.started', globalClock.getTime())
            input_2.reset()
            input_2.setText('')
            # Run 'Begin Routine' code from code_6
            feedback_message = imageName
            print(feedback_message)
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            # keep track of which components have finished
            recall2Components = [recall_text_2, input_2, end_button_2, key_resp_9]
            for thisComponent in recall2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recall_text_2* updates
                
                # if recall_text_2 is starting this frame...
                if recall_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_text_2.frameNStart = frameN  # exact frame index
                    recall_text_2.tStart = t  # local t and not account for scr refresh
                    recall_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_text_2.started')
                    # update status
                    recall_text_2.status = STARTED
                    recall_text_2.setAutoDraw(True)
                
                # if recall_text_2 is active this frame...
                if recall_text_2.status == STARTED:
                    # update params
                    pass
                
                # *input_2* updates
                
                # if input_2 is starting this frame...
                if input_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    input_2.frameNStart = frameN  # exact frame index
                    input_2.tStart = t  # local t and not account for scr refresh
                    input_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(input_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'input_2.started')
                    # update status
                    input_2.status = STARTED
                    input_2.setAutoDraw(True)
                
                # if input_2 is active this frame...
                if input_2.status == STARTED:
                    # update params
                    pass
                
                # *end_button_2* updates
                
                # if end_button_2 is starting this frame...
                if end_button_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_button_2.frameNStart = frameN  # exact frame index
                    end_button_2.tStart = t  # local t and not account for scr refresh
                    end_button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_button_2.started')
                    # update status
                    end_button_2.status = STARTED
                    end_button_2.setAutoDraw(True)
                
                # if end_button_2 is active this frame...
                if end_button_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall2" ---
            for thisComponent in recall2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('recall2.stopped', globalClock.getTime())
            trials_2.addData('input_2.text',input_2.text)
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            trials_2.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                trials_2.addData('key_resp_9.rt', key_resp_9.rt)
                trials_2.addData('key_resp_9.duration', key_resp_9.duration)
            # the Routine "recall2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation_2.started', globalClock.getTime())
            # Run 'Begin Routine' code from code2_2
            response = input_2.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            print(feedbackMessage)
            # keep track of which components have finished
            validation_2Components = []
            for thisComponent in validation_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validation_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation_2" ---
            for thisComponent in validation_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation_2.stopped', globalClock.getTime())
            # the Routine "validation_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_2'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_odd.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_odd.addData('key_resp_14.rt', key_resp_14.rt)
            trials_odd.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split3.started', globalClock.getTime())
        key_resp_17.keys = []
        key_resp_17.rt = []
        _key_resp_17_allKeys = []
        # keep track of which components have finished
        split3Components = [split3_text, key_resp_17]
        for thisComponent in split3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *split3_text* updates
            
            # if split3_text is starting this frame...
            if split3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                split3_text.frameNStart = frameN  # exact frame index
                split3_text.tStart = t  # local t and not account for scr refresh
                split3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(split3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'split3_text.started')
                # update status
                split3_text.status = STARTED
                split3_text.setAutoDraw(True)
            
            # if split3_text is active this frame...
            if split3_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_17* updates
            waitOnFlip = False
            
            # if key_resp_17 is starting this frame...
            if key_resp_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_17.frameNStart = frameN  # exact frame index
                key_resp_17.tStart = t  # local t and not account for scr refresh
                key_resp_17.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_17, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_17.started')
                # update status
                key_resp_17.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_17.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_17.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_17.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_17.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_17_allKeys.extend(theseKeys)
                if len(_key_resp_17_allKeys):
                    key_resp_17.keys = _key_resp_17_allKeys[-1].name  # just the last key pressed
                    key_resp_17.rt = _key_resp_17_allKeys[-1].rt
                    key_resp_17.duration = _key_resp_17_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split3" ---
        for thisComponent in split3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split3.stopped', globalClock.getTime())
        # check responses
        if key_resp_17.keys in ['', [], None]:  # No response was made
            key_resp_17.keys = None
        trials_odd.addData('key_resp_17.keys',key_resp_17.keys)
        if key_resp_17.keys != None:  # we had a response
            trials_odd.addData('key_resp_17.rt', key_resp_17.rt)
            trials_odd.addData('key_resp_17.duration', key_resp_17.duration)
        # the Routine "split3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_4 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_weekD.xlsx'),
            seed=None, name='trials_4')
        thisExp.addLoop(trials_4)  # add the loop to the experiment
        thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        for thisTrial_4 in trials_4:
            currentLoop = trials_4
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
            if thisTrial_4 != None:
                for paramName in thisTrial_4:
                    globals()[paramName] = thisTrial_4[paramName]
            
            # --- Prepare to start Routine "random_position" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('random_position.started', globalClock.getTime())
            # keep track of which components have finished
            random_positionComponents = []
            for thisComponent in random_positionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "random_position" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from rand_pos
                import random
                
                random_number = random.randint(0, 2)
                print("random number = ", random_number)
                
                if(random_number == 1):
                    pos = (-.15, 0)
                elif(random_number == 2):
                    pos = (.15,0)
                else:
                    pos = (0,0)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in random_positionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "random_position" ---
            for thisComponent in random_positionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('random_position.stopped', globalClock.getTime())
            # the Routine "random_position" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "image_display2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('image_display2.started', globalClock.getTime())
            image_weekD.setPos(pos)
            image_weekD.setImage(imageName)
            response_key_weekD.keys = []
            response_key_weekD.rt = []
            _response_key_weekD_allKeys = []
            # keep track of which components have finished
            image_display2Components = [image_weekD, response_key_weekD]
            for thisComponent in image_display2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "image_display2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_weekD* updates
                
                # if image_weekD is starting this frame...
                if image_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_weekD.frameNStart = frameN  # exact frame index
                    image_weekD.tStart = t  # local t and not account for scr refresh
                    image_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_weekD.started')
                    # update status
                    image_weekD.status = STARTED
                    image_weekD.setAutoDraw(True)
                
                # if image_weekD is active this frame...
                if image_weekD.status == STARTED:
                    # update params
                    pass
                
                # *response_key_weekD* updates
                waitOnFlip = False
                
                # if response_key_weekD is starting this frame...
                if response_key_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_weekD.frameNStart = frameN  # exact frame index
                    response_key_weekD.tStart = t  # local t and not account for scr refresh
                    response_key_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_weekD.started')
                    # update status
                    response_key_weekD.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_weekD.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_weekD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_weekD.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_weekD.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_weekD_allKeys.extend(theseKeys)
                    if len(_response_key_weekD_allKeys):
                        response_key_weekD.keys = _response_key_weekD_allKeys[-1].name  # just the last key pressed
                        response_key_weekD.rt = _response_key_weekD_allKeys[-1].rt
                        response_key_weekD.duration = _response_key_weekD_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_display2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_display2" ---
            for thisComponent in image_display2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('image_display2.stopped', globalClock.getTime())
            # check responses
            if response_key_weekD.keys in ['', [], None]:  # No response was made
                response_key_weekD.keys = None
            trials_4.addData('response_key_weekD.keys',response_key_weekD.keys)
            if response_key_weekD.keys != None:  # we had a response
                trials_4.addData('response_key_weekD.rt', response_key_weekD.rt)
                trials_4.addData('response_key_weekD.duration', response_key_weekD.duration)
            # the Routine "image_display2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "recall2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('recall2.started', globalClock.getTime())
            input_2.reset()
            input_2.setText('')
            # Run 'Begin Routine' code from code_6
            feedback_message = imageName
            print(feedback_message)
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            # keep track of which components have finished
            recall2Components = [recall_text_2, input_2, end_button_2, key_resp_9]
            for thisComponent in recall2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recall_text_2* updates
                
                # if recall_text_2 is starting this frame...
                if recall_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_text_2.frameNStart = frameN  # exact frame index
                    recall_text_2.tStart = t  # local t and not account for scr refresh
                    recall_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_text_2.started')
                    # update status
                    recall_text_2.status = STARTED
                    recall_text_2.setAutoDraw(True)
                
                # if recall_text_2 is active this frame...
                if recall_text_2.status == STARTED:
                    # update params
                    pass
                
                # *input_2* updates
                
                # if input_2 is starting this frame...
                if input_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    input_2.frameNStart = frameN  # exact frame index
                    input_2.tStart = t  # local t and not account for scr refresh
                    input_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(input_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'input_2.started')
                    # update status
                    input_2.status = STARTED
                    input_2.setAutoDraw(True)
                
                # if input_2 is active this frame...
                if input_2.status == STARTED:
                    # update params
                    pass
                
                # *end_button_2* updates
                
                # if end_button_2 is starting this frame...
                if end_button_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_button_2.frameNStart = frameN  # exact frame index
                    end_button_2.tStart = t  # local t and not account for scr refresh
                    end_button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_button_2.started')
                    # update status
                    end_button_2.status = STARTED
                    end_button_2.setAutoDraw(True)
                
                # if end_button_2 is active this frame...
                if end_button_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall2" ---
            for thisComponent in recall2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('recall2.stopped', globalClock.getTime())
            trials_4.addData('input_2.text',input_2.text)
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            trials_4.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                trials_4.addData('key_resp_9.rt', key_resp_9.rt)
                trials_4.addData('key_resp_9.duration', key_resp_9.duration)
            # the Routine "recall2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation_2.started', globalClock.getTime())
            # Run 'Begin Routine' code from code2_2
            response = input_2.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            print(feedbackMessage)
            # keep track of which components have finished
            validation_2Components = []
            for thisComponent in validation_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validation_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation_2" ---
            for thisComponent in validation_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation_2.stopped', globalClock.getTime())
            # the Routine "validation_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_4'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed oddParticipant repeats of 'trials_odd'
    
    
    # set up handler to look after randomisation of conditions etc
    trials_even = data.TrialHandler(nReps=evenParticipant, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_even')
    thisExp.addLoop(trials_even)  # add the loop to the experiment
    thisTrials_even = trials_even.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_even.rgb)
    if thisTrials_even != None:
        for paramName in thisTrials_even:
            globals()[paramName] = thisTrials_even[paramName]
    
    for thisTrials_even in trials_even:
        currentLoop = trials_even
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_even.rgb)
        if thisTrials_even != None:
            for paramName in thisTrials_even:
                globals()[paramName] = thisTrials_even[paramName]
        
        # --- Prepare to start Routine "preparation_phase2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase2.started', globalClock.getTime())
        key_resp_7.keys = []
        key_resp_7.rt = []
        _key_resp_7_allKeys = []
        # keep track of which components have finished
        preparation_phase2Components = [preparation_text_2, key_resp_7]
        for thisComponent in preparation_phase2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *preparation_text_2* updates
            
            # if preparation_text_2 is starting this frame...
            if preparation_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preparation_text_2.frameNStart = frameN  # exact frame index
                preparation_text_2.tStart = t  # local t and not account for scr refresh
                preparation_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preparation_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'preparation_text_2.started')
                # update status
                preparation_text_2.status = STARTED
                preparation_text_2.setAutoDraw(True)
            
            # if preparation_text_2 is active this frame...
            if preparation_text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_7* updates
            waitOnFlip = False
            
            # if key_resp_7 is starting this frame...
            if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_7.frameNStart = frameN  # exact frame index
                key_resp_7.tStart = t  # local t and not account for scr refresh
                key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_7.started')
                # update status
                key_resp_7.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_7.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_7_allKeys.extend(theseKeys)
                if len(_key_resp_7_allKeys):
                    key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                    key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                    key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase2" ---
        for thisComponent in preparation_phase2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase2.stopped', globalClock.getTime())
        # check responses
        if key_resp_7.keys in ['', [], None]:  # No response was made
            key_resp_7.keys = None
        trials_even.addData('key_resp_7.keys',key_resp_7.keys)
        if key_resp_7.keys != None:  # we had a response
            trials_even.addData('key_resp_7.rt', key_resp_7.rt)
            trials_even.addData('key_resp_7.duration', key_resp_7.duration)
        # the Routine "preparation_phase2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase2_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase2_2.started', globalClock.getTime())
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # keep track of which components have finished
        preparation_phase2_2Components = [image_4, key_resp_4]
        for thisComponent in preparation_phase2_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase2_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_4* updates
            
            # if image_4 is starting this frame...
            if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_4.frameNStart = frameN  # exact frame index
                image_4.tStart = t  # local t and not account for scr refresh
                image_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_4.started')
                # update status
                image_4.status = STARTED
                image_4.setAutoDraw(True)
            
            # if image_4 is active this frame...
            if image_4.status == STARTED:
                # update params
                pass
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase2_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase2_2" ---
        for thisComponent in preparation_phase2_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase2_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        trials_even.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            trials_even.addData('key_resp_4.rt', key_resp_4.rt)
            trials_even.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "preparation_phase2_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase3.started', globalClock.getTime())
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # keep track of which components have finished
        preparation_phase3Components = [text_2, key_resp_2]
        for thisComponent in preparation_phase3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase3" ---
        for thisComponent in preparation_phase3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase3.stopped', globalClock.getTime())
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        trials_even.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            trials_even.addData('key_resp_2.rt', key_resp_2.rt)
            trials_even.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "preparation_phase3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase4.started', globalClock.getTime())
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # keep track of which components have finished
        preparation_phase4Components = [note_text, note_desc, key_resp_13, press_space_text_4]
        for thisComponent in preparation_phase4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *note_text* updates
            
            # if note_text is starting this frame...
            if note_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_text.frameNStart = frameN  # exact frame index
                note_text.tStart = t  # local t and not account for scr refresh
                note_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_text.started')
                # update status
                note_text.status = STARTED
                note_text.setAutoDraw(True)
            
            # if note_text is active this frame...
            if note_text.status == STARTED:
                # update params
                pass
            
            # *note_desc* updates
            
            # if note_desc is starting this frame...
            if note_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_desc.frameNStart = frameN  # exact frame index
                note_desc.tStart = t  # local t and not account for scr refresh
                note_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_desc.started')
                # update status
                note_desc.status = STARTED
                note_desc.setAutoDraw(True)
            
            # if note_desc is active this frame...
            if note_desc.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *press_space_text_4* updates
            
            # if press_space_text_4 is starting this frame...
            if press_space_text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                press_space_text_4.frameNStart = frameN  # exact frame index
                press_space_text_4.tStart = t  # local t and not account for scr refresh
                press_space_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_space_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_space_text_4.started')
                # update status
                press_space_text_4.status = STARTED
                press_space_text_4.setAutoDraw(True)
            
            # if press_space_text_4 is active this frame...
            if press_space_text_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase4" ---
        for thisComponent in preparation_phase4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase4.stopped', globalClock.getTime())
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials_even.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials_even.addData('key_resp_13.rt', key_resp_13.rt)
            trials_even.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "preparation_phase4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials2_2 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_weekD.xlsx'),
            seed=None, name='trials2_2')
        thisExp.addLoop(trials2_2)  # add the loop to the experiment
        thisTrials2_2 = trials2_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials2_2.rgb)
        if thisTrials2_2 != None:
            for paramName in thisTrials2_2:
                globals()[paramName] = thisTrials2_2[paramName]
        
        for thisTrials2_2 in trials2_2:
            currentLoop = trials2_2
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrials2_2.rgb)
            if thisTrials2_2 != None:
                for paramName in thisTrials2_2:
                    globals()[paramName] = thisTrials2_2[paramName]
            
            # --- Prepare to start Routine "image_display2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('image_display2.started', globalClock.getTime())
            image_weekD.setPos(pos)
            image_weekD.setImage(imageName)
            response_key_weekD.keys = []
            response_key_weekD.rt = []
            _response_key_weekD_allKeys = []
            # keep track of which components have finished
            image_display2Components = [image_weekD, response_key_weekD]
            for thisComponent in image_display2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "image_display2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_weekD* updates
                
                # if image_weekD is starting this frame...
                if image_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_weekD.frameNStart = frameN  # exact frame index
                    image_weekD.tStart = t  # local t and not account for scr refresh
                    image_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_weekD.started')
                    # update status
                    image_weekD.status = STARTED
                    image_weekD.setAutoDraw(True)
                
                # if image_weekD is active this frame...
                if image_weekD.status == STARTED:
                    # update params
                    pass
                
                # *response_key_weekD* updates
                waitOnFlip = False
                
                # if response_key_weekD is starting this frame...
                if response_key_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_weekD.frameNStart = frameN  # exact frame index
                    response_key_weekD.tStart = t  # local t and not account for scr refresh
                    response_key_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_weekD.started')
                    # update status
                    response_key_weekD.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_weekD.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_weekD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_weekD.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_weekD.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_weekD_allKeys.extend(theseKeys)
                    if len(_response_key_weekD_allKeys):
                        response_key_weekD.keys = _response_key_weekD_allKeys[-1].name  # just the last key pressed
                        response_key_weekD.rt = _response_key_weekD_allKeys[-1].rt
                        response_key_weekD.duration = _response_key_weekD_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_display2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_display2" ---
            for thisComponent in image_display2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('image_display2.stopped', globalClock.getTime())
            # check responses
            if response_key_weekD.keys in ['', [], None]:  # No response was made
                response_key_weekD.keys = None
            trials2_2.addData('response_key_weekD.keys',response_key_weekD.keys)
            if response_key_weekD.keys != None:  # we had a response
                trials2_2.addData('response_key_weekD.rt', response_key_weekD.rt)
                trials2_2.addData('response_key_weekD.duration', response_key_weekD.duration)
            # the Routine "image_display2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "recall2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('recall2.started', globalClock.getTime())
            input_2.reset()
            input_2.setText('')
            # Run 'Begin Routine' code from code_6
            feedback_message = imageName
            print(feedback_message)
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            # keep track of which components have finished
            recall2Components = [recall_text_2, input_2, end_button_2, key_resp_9]
            for thisComponent in recall2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recall_text_2* updates
                
                # if recall_text_2 is starting this frame...
                if recall_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_text_2.frameNStart = frameN  # exact frame index
                    recall_text_2.tStart = t  # local t and not account for scr refresh
                    recall_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_text_2.started')
                    # update status
                    recall_text_2.status = STARTED
                    recall_text_2.setAutoDraw(True)
                
                # if recall_text_2 is active this frame...
                if recall_text_2.status == STARTED:
                    # update params
                    pass
                
                # *input_2* updates
                
                # if input_2 is starting this frame...
                if input_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    input_2.frameNStart = frameN  # exact frame index
                    input_2.tStart = t  # local t and not account for scr refresh
                    input_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(input_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'input_2.started')
                    # update status
                    input_2.status = STARTED
                    input_2.setAutoDraw(True)
                
                # if input_2 is active this frame...
                if input_2.status == STARTED:
                    # update params
                    pass
                
                # *end_button_2* updates
                
                # if end_button_2 is starting this frame...
                if end_button_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_button_2.frameNStart = frameN  # exact frame index
                    end_button_2.tStart = t  # local t and not account for scr refresh
                    end_button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_button_2.started')
                    # update status
                    end_button_2.status = STARTED
                    end_button_2.setAutoDraw(True)
                
                # if end_button_2 is active this frame...
                if end_button_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall2" ---
            for thisComponent in recall2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('recall2.stopped', globalClock.getTime())
            trials2_2.addData('input_2.text',input_2.text)
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            trials2_2.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                trials2_2.addData('key_resp_9.rt', key_resp_9.rt)
                trials2_2.addData('key_resp_9.duration', key_resp_9.duration)
            # the Routine "recall2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation_2.started', globalClock.getTime())
            # Run 'Begin Routine' code from code2_2
            response = input_2.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            print(feedbackMessage)
            # keep track of which components have finished
            validation_2Components = []
            for thisComponent in validation_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validation_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation_2" ---
            for thisComponent in validation_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation_2.stopped', globalClock.getTime())
            # the Routine "validation_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials2_2'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_even.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_even.addData('key_resp_14.rt', key_resp_14.rt)
            trials_even.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split1.started', globalClock.getTime())
        key_resp_15.keys = []
        key_resp_15.rt = []
        _key_resp_15_allKeys = []
        # keep track of which components have finished
        split1Components = [text_6, key_resp_15]
        for thisComponent in split1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # *key_resp_15* updates
            waitOnFlip = False
            
            # if key_resp_15 is starting this frame...
            if key_resp_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_15.frameNStart = frameN  # exact frame index
                key_resp_15.tStart = t  # local t and not account for scr refresh
                key_resp_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_15.started')
                # update status
                key_resp_15.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_15.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_15.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_15.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_15.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_15_allKeys.extend(theseKeys)
                if len(_key_resp_15_allKeys):
                    key_resp_15.keys = _key_resp_15_allKeys[-1].name  # just the last key pressed
                    key_resp_15.rt = _key_resp_15_allKeys[-1].rt
                    key_resp_15.duration = _key_resp_15_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split1" ---
        for thisComponent in split1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split1.stopped', globalClock.getTime())
        # check responses
        if key_resp_15.keys in ['', [], None]:  # No response was made
            key_resp_15.keys = None
        trials_even.addData('key_resp_15.keys',key_resp_15.keys)
        if key_resp_15.keys != None:  # we had a response
            trials_even.addData('key_resp_15.rt', key_resp_15.rt)
            trials_even.addData('key_resp_15.duration', key_resp_15.duration)
        # the Routine "split1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_5 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_weekD.xlsx'),
            seed=None, name='trials_5')
        thisExp.addLoop(trials_5)  # add the loop to the experiment
        thisTrial_5 = trials_5.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
        if thisTrial_5 != None:
            for paramName in thisTrial_5:
                globals()[paramName] = thisTrial_5[paramName]
        
        for thisTrial_5 in trials_5:
            currentLoop = trials_5
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_5.rgb)
            if thisTrial_5 != None:
                for paramName in thisTrial_5:
                    globals()[paramName] = thisTrial_5[paramName]
            
            # --- Prepare to start Routine "image_display2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('image_display2.started', globalClock.getTime())
            image_weekD.setPos(pos)
            image_weekD.setImage(imageName)
            response_key_weekD.keys = []
            response_key_weekD.rt = []
            _response_key_weekD_allKeys = []
            # keep track of which components have finished
            image_display2Components = [image_weekD, response_key_weekD]
            for thisComponent in image_display2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "image_display2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_weekD* updates
                
                # if image_weekD is starting this frame...
                if image_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_weekD.frameNStart = frameN  # exact frame index
                    image_weekD.tStart = t  # local t and not account for scr refresh
                    image_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_weekD.started')
                    # update status
                    image_weekD.status = STARTED
                    image_weekD.setAutoDraw(True)
                
                # if image_weekD is active this frame...
                if image_weekD.status == STARTED:
                    # update params
                    pass
                
                # *response_key_weekD* updates
                waitOnFlip = False
                
                # if response_key_weekD is starting this frame...
                if response_key_weekD.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_weekD.frameNStart = frameN  # exact frame index
                    response_key_weekD.tStart = t  # local t and not account for scr refresh
                    response_key_weekD.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_weekD, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_weekD.started')
                    # update status
                    response_key_weekD.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_weekD.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_weekD.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_weekD.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_weekD.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_weekD_allKeys.extend(theseKeys)
                    if len(_response_key_weekD_allKeys):
                        response_key_weekD.keys = _response_key_weekD_allKeys[-1].name  # just the last key pressed
                        response_key_weekD.rt = _response_key_weekD_allKeys[-1].rt
                        response_key_weekD.duration = _response_key_weekD_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_display2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_display2" ---
            for thisComponent in image_display2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('image_display2.stopped', globalClock.getTime())
            # check responses
            if response_key_weekD.keys in ['', [], None]:  # No response was made
                response_key_weekD.keys = None
            trials_5.addData('response_key_weekD.keys',response_key_weekD.keys)
            if response_key_weekD.keys != None:  # we had a response
                trials_5.addData('response_key_weekD.rt', response_key_weekD.rt)
                trials_5.addData('response_key_weekD.duration', response_key_weekD.duration)
            # the Routine "image_display2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "recall2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('recall2.started', globalClock.getTime())
            input_2.reset()
            input_2.setText('')
            # Run 'Begin Routine' code from code_6
            feedback_message = imageName
            print(feedback_message)
            key_resp_9.keys = []
            key_resp_9.rt = []
            _key_resp_9_allKeys = []
            # keep track of which components have finished
            recall2Components = [recall_text_2, input_2, end_button_2, key_resp_9]
            for thisComponent in recall2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "recall2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recall_text_2* updates
                
                # if recall_text_2 is starting this frame...
                if recall_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recall_text_2.frameNStart = frameN  # exact frame index
                    recall_text_2.tStart = t  # local t and not account for scr refresh
                    recall_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recall_text_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_text_2.started')
                    # update status
                    recall_text_2.status = STARTED
                    recall_text_2.setAutoDraw(True)
                
                # if recall_text_2 is active this frame...
                if recall_text_2.status == STARTED:
                    # update params
                    pass
                
                # *input_2* updates
                
                # if input_2 is starting this frame...
                if input_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    input_2.frameNStart = frameN  # exact frame index
                    input_2.tStart = t  # local t and not account for scr refresh
                    input_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(input_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'input_2.started')
                    # update status
                    input_2.status = STARTED
                    input_2.setAutoDraw(True)
                
                # if input_2 is active this frame...
                if input_2.status == STARTED:
                    # update params
                    pass
                
                # *end_button_2* updates
                
                # if end_button_2 is starting this frame...
                if end_button_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    end_button_2.frameNStart = frameN  # exact frame index
                    end_button_2.tStart = t  # local t and not account for scr refresh
                    end_button_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(end_button_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_button_2.started')
                    # update status
                    end_button_2.status = STARTED
                    end_button_2.setAutoDraw(True)
                
                # if end_button_2 is active this frame...
                if end_button_2.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_9* updates
                waitOnFlip = False
                
                # if key_resp_9 is starting this frame...
                if key_resp_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_9.frameNStart = frameN  # exact frame index
                    key_resp_9.tStart = t  # local t and not account for scr refresh
                    key_resp_9.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_9, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_9.started')
                    # update status
                    key_resp_9.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_9.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_9.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_9.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_9.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_9_allKeys.extend(theseKeys)
                    if len(_key_resp_9_allKeys):
                        key_resp_9.keys = _key_resp_9_allKeys[-1].name  # just the last key pressed
                        key_resp_9.rt = _key_resp_9_allKeys[-1].rt
                        key_resp_9.duration = _key_resp_9_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in recall2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "recall2" ---
            for thisComponent in recall2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('recall2.stopped', globalClock.getTime())
            trials_5.addData('input_2.text',input_2.text)
            # check responses
            if key_resp_9.keys in ['', [], None]:  # No response was made
                key_resp_9.keys = None
            trials_5.addData('key_resp_9.keys',key_resp_9.keys)
            if key_resp_9.keys != None:  # we had a response
                trials_5.addData('key_resp_9.rt', key_resp_9.rt)
                trials_5.addData('key_resp_9.duration', key_resp_9.duration)
            # the Routine "recall2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation_2.started', globalClock.getTime())
            # Run 'Begin Routine' code from code2_2
            response = input_2.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            print(feedbackMessage)
            # keep track of which components have finished
            validation_2Components = []
            for thisComponent in validation_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validation_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation_2" ---
            for thisComponent in validation_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation_2.stopped', globalClock.getTime())
            # the Routine "validation_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_5'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_even.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_even.addData('key_resp_14.rt', key_resp_14.rt)
            trials_even.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split2.started', globalClock.getTime())
        key_resp_16.keys = []
        key_resp_16.rt = []
        _key_resp_16_allKeys = []
        # keep track of which components have finished
        split2Components = [text_9, key_resp_16]
        for thisComponent in split2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # *key_resp_16* updates
            waitOnFlip = False
            
            # if key_resp_16 is starting this frame...
            if key_resp_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_16.frameNStart = frameN  # exact frame index
                key_resp_16.tStart = t  # local t and not account for scr refresh
                key_resp_16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_16.started')
                # update status
                key_resp_16.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_16.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_16.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_16.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_16.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_16_allKeys.extend(theseKeys)
                if len(_key_resp_16_allKeys):
                    key_resp_16.keys = _key_resp_16_allKeys[-1].name  # just the last key pressed
                    key_resp_16.rt = _key_resp_16_allKeys[-1].rt
                    key_resp_16.duration = _key_resp_16_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split2" ---
        for thisComponent in split2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split2.stopped', globalClock.getTime())
        # check responses
        if key_resp_16.keys in ['', [], None]:  # No response was made
            key_resp_16.keys = None
        trials_even.addData('key_resp_16.keys',key_resp_16.keys)
        if key_resp_16.keys != None:  # we had a response
            trials_even.addData('key_resp_16.rt', key_resp_16.rt)
            trials_even.addData('key_resp_16.duration', key_resp_16.duration)
        # the Routine "split2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Preparation_Phase1_3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Preparation_Phase1_3.started', globalClock.getTime())
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        Preparation_Phase1_3Components = [text, key_resp]
        for thisComponent in Preparation_Phase1_3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Preparation_Phase1_3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preparation_Phase1_3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preparation_Phase1_3" ---
        for thisComponent in Preparation_Phase1_3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Preparation_Phase1_3.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_even.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_even.addData('key_resp.rt', key_resp.rt)
            trials_even.addData('key_resp.duration', key_resp.duration)
        # the Routine "Preparation_Phase1_3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Preperation_Phase2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('Preperation_Phase2.started', globalClock.getTime())
        preparation_key.keys = []
        preparation_key.rt = []
        _preparation_key_allKeys = []
        # keep track of which components have finished
        Preperation_Phase2Components = [image_3, preparation_key]
        for thisComponent in Preperation_Phase2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Preperation_Phase2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_3* updates
            
            # if image_3 is starting this frame...
            if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_3.frameNStart = frameN  # exact frame index
                image_3.tStart = t  # local t and not account for scr refresh
                image_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_3.started')
                # update status
                image_3.status = STARTED
                image_3.setAutoDraw(True)
            
            # if image_3 is active this frame...
            if image_3.status == STARTED:
                # update params
                pass
            
            # if image_3 is stopping this frame...
            if image_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_3.tStartRefresh + 120-frameTolerance:
                    # keep track of stop time/frame for later
                    image_3.tStop = t  # not accounting for scr refresh
                    image_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.stopped')
                    # update status
                    image_3.status = FINISHED
                    image_3.setAutoDraw(False)
            
            # *preparation_key* updates
            waitOnFlip = False
            
            # if preparation_key is starting this frame...
            if preparation_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preparation_key.frameNStart = frameN  # exact frame index
                preparation_key.tStart = t  # local t and not account for scr refresh
                preparation_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preparation_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'preparation_key.started')
                # update status
                preparation_key.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(preparation_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(preparation_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if preparation_key.status == STARTED and not waitOnFlip:
                theseKeys = preparation_key.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _preparation_key_allKeys.extend(theseKeys)
                if len(_preparation_key_allKeys):
                    preparation_key.keys = _preparation_key_allKeys[-1].name  # just the last key pressed
                    preparation_key.rt = _preparation_key_allKeys[-1].rt
                    preparation_key.duration = _preparation_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Preperation_Phase2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Preperation_Phase2" ---
        for thisComponent in Preperation_Phase2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('Preperation_Phase2.stopped', globalClock.getTime())
        # check responses
        if preparation_key.keys in ['', [], None]:  # No response was made
            preparation_key.keys = None
        trials_even.addData('preparation_key.keys',preparation_key.keys)
        if preparation_key.keys != None:  # we had a response
            trials_even.addData('preparation_key.rt', preparation_key.rt)
            trials_even.addData('preparation_key.duration', preparation_key.duration)
        # the Routine "Preperation_Phase2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase3_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase3_2.started', globalClock.getTime())
        key_resp_5.keys = []
        key_resp_5.rt = []
        _key_resp_5_allKeys = []
        # keep track of which components have finished
        preparation_phase3_2Components = [memorization_text, key_resp_5]
        for thisComponent in preparation_phase3_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase3_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *memorization_text* updates
            
            # if memorization_text is starting this frame...
            if memorization_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                memorization_text.frameNStart = frameN  # exact frame index
                memorization_text.tStart = t  # local t and not account for scr refresh
                memorization_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(memorization_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'memorization_text.started')
                # update status
                memorization_text.status = STARTED
                memorization_text.setAutoDraw(True)
            
            # if memorization_text is active this frame...
            if memorization_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_5* updates
            waitOnFlip = False
            
            # if key_resp_5 is starting this frame...
            if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_5.frameNStart = frameN  # exact frame index
                key_resp_5.tStart = t  # local t and not account for scr refresh
                key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_5.started')
                # update status
                key_resp_5.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_5.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_5.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_5_allKeys.extend(theseKeys)
                if len(_key_resp_5_allKeys):
                    key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                    key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                    key_resp_5.duration = _key_resp_5_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase3_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase3_2" ---
        for thisComponent in preparation_phase3_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase3_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_5.keys in ['', [], None]:  # No response was made
            key_resp_5.keys = None
        trials_even.addData('key_resp_5.keys',key_resp_5.keys)
        if key_resp_5.keys != None:  # we had a response
            trials_even.addData('key_resp_5.rt', key_resp_5.rt)
            trials_even.addData('key_resp_5.duration', key_resp_5.duration)
        # the Routine "preparation_phase3_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "preparation_phase4" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('preparation_phase4.started', globalClock.getTime())
        key_resp_13.keys = []
        key_resp_13.rt = []
        _key_resp_13_allKeys = []
        # keep track of which components have finished
        preparation_phase4Components = [note_text, note_desc, key_resp_13, press_space_text_4]
        for thisComponent in preparation_phase4Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "preparation_phase4" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *note_text* updates
            
            # if note_text is starting this frame...
            if note_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_text.frameNStart = frameN  # exact frame index
                note_text.tStart = t  # local t and not account for scr refresh
                note_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_text.started')
                # update status
                note_text.status = STARTED
                note_text.setAutoDraw(True)
            
            # if note_text is active this frame...
            if note_text.status == STARTED:
                # update params
                pass
            
            # *note_desc* updates
            
            # if note_desc is starting this frame...
            if note_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                note_desc.frameNStart = frameN  # exact frame index
                note_desc.tStart = t  # local t and not account for scr refresh
                note_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(note_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'note_desc.started')
                # update status
                note_desc.status = STARTED
                note_desc.setAutoDraw(True)
            
            # if note_desc is active this frame...
            if note_desc.status == STARTED:
                # update params
                pass
            
            # *key_resp_13* updates
            waitOnFlip = False
            
            # if key_resp_13 is starting this frame...
            if key_resp_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_13.frameNStart = frameN  # exact frame index
                key_resp_13.tStart = t  # local t and not account for scr refresh
                key_resp_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_13.started')
                # update status
                key_resp_13.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_13.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_13.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_13.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_13.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_13_allKeys.extend(theseKeys)
                if len(_key_resp_13_allKeys):
                    key_resp_13.keys = _key_resp_13_allKeys[-1].name  # just the last key pressed
                    key_resp_13.rt = _key_resp_13_allKeys[-1].rt
                    key_resp_13.duration = _key_resp_13_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *press_space_text_4* updates
            
            # if press_space_text_4 is starting this frame...
            if press_space_text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                press_space_text_4.frameNStart = frameN  # exact frame index
                press_space_text_4.tStart = t  # local t and not account for scr refresh
                press_space_text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(press_space_text_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'press_space_text_4.started')
                # update status
                press_space_text_4.status = STARTED
                press_space_text_4.setAutoDraw(True)
            
            # if press_space_text_4 is active this frame...
            if press_space_text_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in preparation_phase4Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "preparation_phase4" ---
        for thisComponent in preparation_phase4Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('preparation_phase4.stopped', globalClock.getTime())
        # check responses
        if key_resp_13.keys in ['', [], None]:  # No response was made
            key_resp_13.keys = None
        trials_even.addData('key_resp_13.keys',key_resp_13.keys)
        if key_resp_13.keys != None:  # we had a response
            trials_even.addData('key_resp_13.rt', key_resp_13.rt)
            trials_even.addData('key_resp_13.duration', key_resp_13.duration)
        # the Routine "preparation_phase4" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_3 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_week7.xlsx'),
            seed=None, name='trials_3')
        thisExp.addLoop(trials_3)  # add the loop to the experiment
        thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        for thisTrial_3 in trials_3:
            currentLoop = trials_3
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
            if thisTrial_3 != None:
                for paramName in thisTrial_3:
                    globals()[paramName] = thisTrial_3[paramName]
            
            # --- Prepare to start Routine "newRoutine" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('newRoutine.started', globalClock.getTime())
            image_2.setPos(pos)
            image_2.setImage(imageName)
            response_key_week7.keys = []
            response_key_week7.rt = []
            _response_key_week7_allKeys = []
            # keep track of which components have finished
            newRoutineComponents = [image_2, response_key_week7]
            for thisComponent in newRoutineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "newRoutine" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # *response_key_week7* updates
                waitOnFlip = False
                
                # if response_key_week7 is starting this frame...
                if response_key_week7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_week7.frameNStart = frameN  # exact frame index
                    response_key_week7.tStart = t  # local t and not account for scr refresh
                    response_key_week7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_week7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_week7.started')
                    # update status
                    response_key_week7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_week7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_week7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_week7.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_week7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_week7_allKeys.extend(theseKeys)
                    if len(_response_key_week7_allKeys):
                        response_key_week7.keys = _response_key_week7_allKeys[-1].name  # just the last key pressed
                        response_key_week7.rt = _response_key_week7_allKeys[-1].rt
                        response_key_week7.duration = _response_key_week7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in newRoutineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "newRoutine" ---
            for thisComponent in newRoutineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('newRoutine.stopped', globalClock.getTime())
            # check responses
            if response_key_week7.keys in ['', [], None]:  # No response was made
                response_key_week7.keys = None
            trials_3.addData('response_key_week7.keys',response_key_week7.keys)
            if response_key_week7.keys != None:  # we had a response
                trials_3.addData('response_key_week7.rt', response_key_week7.rt)
                trials_3.addData('response_key_week7.duration', response_key_week7.duration)
            # the Routine "newRoutine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            textbox.reset()
            textbox.setText('')
            # Run 'Begin Routine' code from code_5
            feedback_message = imageName
            print(feedback_message)
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # keep track of which components have finished
            trialComponents = [recallText, textbox, endButton, key_resp_8]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recallText* updates
                
                # if recallText is starting this frame...
                if recallText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recallText.frameNStart = frameN  # exact frame index
                    recallText.tStart = t  # local t and not account for scr refresh
                    recallText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recallText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recallText.started')
                    # update status
                    recallText.status = STARTED
                    recallText.setAutoDraw(True)
                
                # if recallText is active this frame...
                if recallText.status == STARTED:
                    # update params
                    pass
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.started')
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                
                # *endButton* updates
                
                # if endButton is starting this frame...
                if endButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    endButton.frameNStart = frameN  # exact frame index
                    endButton.tStart = t  # local t and not account for scr refresh
                    endButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(endButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'endButton.started')
                    # update status
                    endButton.status = STARTED
                    endButton.setAutoDraw(True)
                
                # if endButton is active this frame...
                if endButton.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            trials_3.addData('textbox.text',textbox.text)
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            trials_3.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                trials_3.addData('key_resp_8.rt', key_resp_8.rt)
                trials_3.addData('key_resp_8.duration', key_resp_8.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_3
            response = textbox.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            
            print("correct ans", correctAns)
            print("response", response)
            print("feedback", feedbackMessage)
            # keep track of which components have finished
            validationComponents = []
            for thisComponent in validationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation" ---
            for thisComponent in validationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation.stopped', globalClock.getTime())
            # the Routine "validation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_3'
        
        
        # --- Prepare to start Routine "break_2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('break_2.started', globalClock.getTime())
        key_resp_14.keys = []
        key_resp_14.rt = []
        _key_resp_14_allKeys = []
        # keep track of which components have finished
        break_2Components = [break_text, key_resp_14, skip_break]
        for thisComponent in break_2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 30.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # if break_text is stopping this frame...
            if break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > break_text.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    break_text.tStop = t  # not accounting for scr refresh
                    break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_text.stopped')
                    # update status
                    break_text.status = FINISHED
                    break_text.setAutoDraw(False)
            
            # *key_resp_14* updates
            waitOnFlip = False
            
            # if key_resp_14 is starting this frame...
            if key_resp_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_14.frameNStart = frameN  # exact frame index
                key_resp_14.tStart = t  # local t and not account for scr refresh
                key_resp_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_14.started')
                # update status
                key_resp_14.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_14.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_14.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp_14 is stopping this frame...
            if key_resp_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp_14.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp_14.tStop = t  # not accounting for scr refresh
                    key_resp_14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_14.stopped')
                    # update status
                    key_resp_14.status = FINISHED
                    key_resp_14.status = FINISHED
            if key_resp_14.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_14.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_14_allKeys.extend(theseKeys)
                if len(_key_resp_14_allKeys):
                    key_resp_14.keys = _key_resp_14_allKeys[-1].name  # just the last key pressed
                    key_resp_14.rt = _key_resp_14_allKeys[-1].rt
                    key_resp_14.duration = _key_resp_14_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *skip_break* updates
            
            # if skip_break is starting this frame...
            if skip_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                skip_break.frameNStart = frameN  # exact frame index
                skip_break.tStart = t  # local t and not account for scr refresh
                skip_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(skip_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'skip_break.started')
                # update status
                skip_break.status = STARTED
                skip_break.setAutoDraw(True)
            
            # if skip_break is active this frame...
            if skip_break.status == STARTED:
                # update params
                pass
            
            # if skip_break is stopping this frame...
            if skip_break.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > skip_break.tStartRefresh + 30-frameTolerance:
                    # keep track of stop time/frame for later
                    skip_break.tStop = t  # not accounting for scr refresh
                    skip_break.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_break.stopped')
                    # update status
                    skip_break.status = FINISHED
                    skip_break.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('break_2.stopped', globalClock.getTime())
        # check responses
        if key_resp_14.keys in ['', [], None]:  # No response was made
            key_resp_14.keys = None
        trials_even.addData('key_resp_14.keys',key_resp_14.keys)
        if key_resp_14.keys != None:  # we had a response
            trials_even.addData('key_resp_14.rt', key_resp_14.rt)
            trials_even.addData('key_resp_14.duration', key_resp_14.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-30.000000)
        
        # --- Prepare to start Routine "split3" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('split3.started', globalClock.getTime())
        key_resp_17.keys = []
        key_resp_17.rt = []
        _key_resp_17_allKeys = []
        # keep track of which components have finished
        split3Components = [split3_text, key_resp_17]
        for thisComponent in split3Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "split3" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *split3_text* updates
            
            # if split3_text is starting this frame...
            if split3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                split3_text.frameNStart = frameN  # exact frame index
                split3_text.tStart = t  # local t and not account for scr refresh
                split3_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(split3_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'split3_text.started')
                # update status
                split3_text.status = STARTED
                split3_text.setAutoDraw(True)
            
            # if split3_text is active this frame...
            if split3_text.status == STARTED:
                # update params
                pass
            
            # *key_resp_17* updates
            waitOnFlip = False
            
            # if key_resp_17 is starting this frame...
            if key_resp_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_17.frameNStart = frameN  # exact frame index
                key_resp_17.tStart = t  # local t and not account for scr refresh
                key_resp_17.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_17, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_17.started')
                # update status
                key_resp_17.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_17.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_17.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_17.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_17.getKeys(keyList=['right','space','return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_17_allKeys.extend(theseKeys)
                if len(_key_resp_17_allKeys):
                    key_resp_17.keys = _key_resp_17_allKeys[-1].name  # just the last key pressed
                    key_resp_17.rt = _key_resp_17_allKeys[-1].rt
                    key_resp_17.duration = _key_resp_17_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in split3Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "split3" ---
        for thisComponent in split3Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('split3.stopped', globalClock.getTime())
        # check responses
        if key_resp_17.keys in ['', [], None]:  # No response was made
            key_resp_17.keys = None
        trials_even.addData('key_resp_17.keys',key_resp_17.keys)
        if key_resp_17.keys != None:  # we had a response
            trials_even.addData('key_resp_17.rt', key_resp_17.rt)
            trials_even.addData('key_resp_17.duration', key_resp_17.duration)
        # the Routine "split3" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_6 = data.TrialHandler(nReps=1.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=data.importConditions('filtered_stimuli_week7.xlsx'),
            seed=None, name='trials_6')
        thisExp.addLoop(trials_6)  # add the loop to the experiment
        thisTrial_6 = trials_6.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_6.rgb)
        if thisTrial_6 != None:
            for paramName in thisTrial_6:
                globals()[paramName] = thisTrial_6[paramName]
        
        for thisTrial_6 in trials_6:
            currentLoop = trials_6
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisTrial_6.rgb)
            if thisTrial_6 != None:
                for paramName in thisTrial_6:
                    globals()[paramName] = thisTrial_6[paramName]
            
            # --- Prepare to start Routine "newRoutine" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('newRoutine.started', globalClock.getTime())
            image_2.setPos(pos)
            image_2.setImage(imageName)
            response_key_week7.keys = []
            response_key_week7.rt = []
            _response_key_week7_allKeys = []
            # keep track of which components have finished
            newRoutineComponents = [image_2, response_key_week7]
            for thisComponent in newRoutineComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "newRoutine" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # *response_key_week7* updates
                waitOnFlip = False
                
                # if response_key_week7 is starting this frame...
                if response_key_week7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    response_key_week7.frameNStart = frameN  # exact frame index
                    response_key_week7.tStart = t  # local t and not account for scr refresh
                    response_key_week7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(response_key_week7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'response_key_week7.started')
                    # update status
                    response_key_week7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(response_key_week7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(response_key_week7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if response_key_week7.status == STARTED and not waitOnFlip:
                    theseKeys = response_key_week7.getKeys(keyList=['space', 'right', 'return'], ignoreKeys=["escape"], waitRelease=False)
                    _response_key_week7_allKeys.extend(theseKeys)
                    if len(_response_key_week7_allKeys):
                        response_key_week7.keys = _response_key_week7_allKeys[-1].name  # just the last key pressed
                        response_key_week7.rt = _response_key_week7_allKeys[-1].rt
                        response_key_week7.duration = _response_key_week7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in newRoutineComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "newRoutine" ---
            for thisComponent in newRoutineComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('newRoutine.stopped', globalClock.getTime())
            # check responses
            if response_key_week7.keys in ['', [], None]:  # No response was made
                response_key_week7.keys = None
            trials_6.addData('response_key_week7.keys',response_key_week7.keys)
            if response_key_week7.keys != None:  # we had a response
                trials_6.addData('response_key_week7.rt', response_key_week7.rt)
                trials_6.addData('response_key_week7.duration', response_key_week7.duration)
            # the Routine "newRoutine" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('trial.started', globalClock.getTime())
            textbox.reset()
            textbox.setText('')
            # Run 'Begin Routine' code from code_5
            feedback_message = imageName
            print(feedback_message)
            key_resp_8.keys = []
            key_resp_8.rt = []
            _key_resp_8_allKeys = []
            # keep track of which components have finished
            trialComponents = [recallText, textbox, endButton, key_resp_8]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *recallText* updates
                
                # if recallText is starting this frame...
                if recallText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    recallText.frameNStart = frameN  # exact frame index
                    recallText.tStart = t  # local t and not account for scr refresh
                    recallText.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(recallText, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recallText.started')
                    # update status
                    recallText.status = STARTED
                    recallText.setAutoDraw(True)
                
                # if recallText is active this frame...
                if recallText.status == STARTED:
                    # update params
                    pass
                
                # *textbox* updates
                
                # if textbox is starting this frame...
                if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    textbox.frameNStart = frameN  # exact frame index
                    textbox.tStart = t  # local t and not account for scr refresh
                    textbox.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.started')
                    # update status
                    textbox.status = STARTED
                    textbox.setAutoDraw(True)
                
                # if textbox is active this frame...
                if textbox.status == STARTED:
                    # update params
                    pass
                
                # *endButton* updates
                
                # if endButton is starting this frame...
                if endButton.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    endButton.frameNStart = frameN  # exact frame index
                    endButton.tStart = t  # local t and not account for scr refresh
                    endButton.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(endButton, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'endButton.started')
                    # update status
                    endButton.status = STARTED
                    endButton.setAutoDraw(True)
                
                # if endButton is active this frame...
                if endButton.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_8* updates
                waitOnFlip = False
                
                # if key_resp_8 is starting this frame...
                if key_resp_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_8.frameNStart = frameN  # exact frame index
                    key_resp_8.tStart = t  # local t and not account for scr refresh
                    key_resp_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_8.started')
                    # update status
                    key_resp_8.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_8.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_8.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_8.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_8.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_8_allKeys.extend(theseKeys)
                    if len(_key_resp_8_allKeys):
                        key_resp_8.keys = _key_resp_8_allKeys[-1].name  # just the last key pressed
                        key_resp_8.rt = _key_resp_8_allKeys[-1].rt
                        key_resp_8.duration = _key_resp_8_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('trial.stopped', globalClock.getTime())
            trials_6.addData('textbox.text',textbox.text)
            # check responses
            if key_resp_8.keys in ['', [], None]:  # No response was made
                key_resp_8.keys = None
            trials_6.addData('key_resp_8.keys',key_resp_8.keys)
            if key_resp_8.keys != None:  # we had a response
                trials_6.addData('key_resp_8.rt', key_resp_8.rt)
                trials_6.addData('key_resp_8.duration', key_resp_8.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "validation" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('validation.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_3
            response = textbox.text
            
            time_correct = 2
            time_incorrect = 1.9
            duration = time_correct
            
            response = response.rstrip('\n')
            response = response.lower()
            
            if len(response) >= 3 and response[:3] == "fry":
                response = "fri" + response[3:]
            if len(response) >= 3 and response[:3] == "wen":
                response = "wed" + response[3:]    
            if len(response) >= 3 and response[:3] == "thr":
                response = "thu" + response[3:]
            if len(response) >= 2 and response[-1] == '8' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '08'
            if len(response) >= 2 and response[-1] == '9' and response[-2] != '0' and response[-2] != '1':
                response = response[:-1] + '09'
            
            if len(response) >= 4 and response[3] == ' ':
                    response = response[:3] + '-' + response[4:]
            elif len(response) > 3 and response[3] != '-':
                response = response[:3] + '-' + response[3:]
            
            correctAns = imageName[13:19]
            if(correctAns == response):
                feedbackMessage = "Correct"
                duration = time_correct
            else:
                feedbackMessage = "Incorrect"
                duration = time_incorrect
            
            print("correct ans", correctAns)
            print("response", response)
            print("feedback", feedbackMessage)
            # keep track of which components have finished
            validationComponents = []
            for thisComponent in validationComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "validation" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in validationComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "validation" ---
            for thisComponent in validationComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('validation.stopped', globalClock.getTime())
            # the Routine "validation" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed 1.0 repeats of 'trials_6'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed evenParticipant repeats of 'trials_even'
    
    
    # --- Prepare to start Routine "End_Screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('End_Screen.started', globalClock.getTime())
    # keep track of which components have finished
    End_ScreenComponents = [text_3]
    for thisComponent in End_ScreenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End_Screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # if text_3 is stopping this frame...
        if text_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_3.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_3.tStop = t  # not accounting for scr refresh
                text_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.stopped')
                # update status
                text_3.status = FINISHED
                text_3.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End_ScreenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End_Screen" ---
    for thisComponent in End_ScreenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('End_Screen.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
