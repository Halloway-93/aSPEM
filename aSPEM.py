#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Using psychopy to perform an experiment on the role of a bias in the direction """

import sys
import os
import numpy as np
import pickle

def binomial_motion(N_trials, N_blocks, tau, seed, Jeffreys=True, N_layer=3):
    """

    about Jeffrey's prior : see wikipedia
    st_dict = dict( fontsize =14, fontstyle = 'italic’) % déjà j’ai pensé à ca :)

    """

    from scipy.stats import beta
    np.random.seed(seed)

    trials = np.arange(N_trials)
    p = np.random.rand(N_trials, N_blocks, N_layer)

    for trial in trials:
        p[trial, :, 2] = np.random.rand(1, N_blocks) < 1/tau # switch
        if Jeffreys: p_random = beta.rvs(a=.5, b=.5, size=N_blocks)
        else: p_random = np.random.rand(1, N_blocks)
        p[trial, :, 1] = (1 - p[trial, :, 2])*p[trial-1, :, 1] + p[trial, :, 2] * p_random # probability
        p[trial, :, 0] =  p[trial, :, 1] > np.random.rand(1, N_blocks) # Bernouilli trial

    return (trials, p)

class aSPEM(object):
    """ docstring for the aSPEM class. """

    def __init__(self, mode, timeStr, observer='test') :
        self.mode = mode
        self.observer = observer
        self.timeStr = str(timeStr)
        self.init()

    def init(self) :

        self.dry_run = True
        self.dry_run = False
        self.experiment = 'aSPEM'
        self.instructions = """ TODO """

        # ---------------------------------------------------
        # setup values
        # ---------------------------------------------------
        cachedir = 'data_cache'
        datadir = 'data'
        import os
        for dir_ in [datadir, cachedir] :
            try: os.mkdir(dir_)
            except: pass

        file = self.mode + '_' + self.observer + '_' + self.timeStr + '.pkl'
        if file in os.listdir(datadir) :
            with open(os.path.join(datadir, file), 'rb') as fichier :
                self.param_exp = pickle.load(fichier, encoding='latin1')

        else :
            expInfo = {"Sujet":'', "Age":''}
            if not self.mode is 'model':
                # Présente un dialogue pour changer les paramètres
                Nom_exp = u'aSPEM'
                try:
                    from psychopy import gui
                    dlg = gui.DlgFromDict(expInfo, title=Nom_exp)
                    PSYCHOPY = True
                except:
                    PSYCHOPY = False

            self.observer = expInfo["Sujet"]
            self.age = expInfo["Age"]

            # width and height of your screen
            screen_width_px = 1440 #1920 #1280 for ordi enregistrement
            screen_height_px = 900 #1080 #1024 for ordi enregistrement
            framerate = 60 #100.for ordi enregistrement
            screen = 0 # 1 pour afficher sur l'écran 2 (ne marche pas pour eyeMvt (mac))

            screen_width_cm = 37 #57. # (cm)
            viewingDistance = 57. # (cm) TODO : what is the equivalent viewing distance?
            screen_width_deg = 2. * np.arctan((screen_width_cm/2) / viewingDistance) * 180/np.pi
            #px_per_deg = screen_height_px / screen_width_deg
            px_per_deg = screen_width_px / screen_width_deg

            # ---------------------------------------------------
            # stimulus parameters
            # ---------------------------------------------------
            dot_size = 10 # (0.02*screen_height_px)
            V_X_deg = 15 #20   # deg/s   # 15 for 'eyeMvt'
            V_X = px_per_deg * V_X_deg     # pixel/s

            RashBass  = 100  # ms - pour reculer la cible à t=0 de sa vitesse * latency=RashBass

            saccade_px = .618*screen_height_px
            offset = 0 #.2*screen_height_px

            # ---------------------------------------------------
            # exploration parameters
            # ---------------------------------------------------
            N_blocks = 3 # 4 blocks avant
            seed = 51 #119 #2017
            N_trials = 200
            tau = N_trials/5.
            (trials, p) = binomial_motion(N_trials, N_blocks, tau=tau, seed=seed, N_layer=3)
            stim_tau = .75 #1 #.35 # in seconds # 1.5 for 'eyeMvt'

            gray_tau = .0 # in seconds
            T =  stim_tau + gray_tau
            N_frame_stim = int(stim_tau*framerate)
            # ---------------------------------------------------

            self.param_exp = dict(N_blocks=N_blocks, seed=seed, N_trials=N_trials, p=p, tau=tau,
                            stim_tau =stim_tau,
                            N_frame_stim=N_frame_stim, T=T,
                            datadir=datadir, cachedir=cachedir,
                            framerate=framerate,
                            screen=screen,
                            screen_width_px=screen_width_px, screen_height_px=screen_height_px,
                            px_per_deg=px_per_deg, offset=offset,
                            dot_size=dot_size, V_X_deg=V_X_deg, V_X =V_X, RashBass=RashBass, saccade_px=saccade_px,
                            mode=self.mode, observer=self.observer, age=self.age, timeStr=self.timeStr)

    def print_protocol(self):
        if True: #try:
            N_blocks = self.param_exp['N_blocks']
            N_trials = self.param_exp['N_trials']
            N_frame_stim = self.param_exp['N_frame_stim']
            T = self.param_exp['T']
            return "TODO"
    #         return """
    # ##########################
    # #  PROTOCOL  #
    # ##########################
    #
        # except:
        #     return 'blurg'


    def exp_name(self):
        return os.path.join(self.param_exp['datadir'], self.mode + '_' + self.observer + '_' + self.timeStr + '.pkl')

    def run_experiment(self, verb=True):

        #if verb: print('launching experiment')

        from psychopy import visual, core, event, logging, prefs
        prefs.general['audioLib'] = [u'pygame']
        from psychopy import sound

        if self.mode=='eyeMvt' :
            import EyeTracking as ET
            ET = ET.EyeTracking(self.param_exp['screen_width_px'], self.param_exp['screen_height_px'], self.param_exp['dot_size'], self.param_exp['N_trials'], self.observer, self.param_exp['datadir'], self.timeStr)

        #logging.console.setLevel(logging.WARNING)
        #if verb: print('launching experiment')
        #logging.console.setLevel(logging.WARNING)
        #if verb: print('go!')

        # ---------------------------------------------------
        win = visual.Window([self.param_exp['screen_width_px'], self.param_exp['screen_height_px']], color=(0, 0, 0),
                            allowGUI=False, fullscr=True, screen=self.param_exp['screen'], units='pix') # enlever fullscr=True pour écran 2

        win.setRecordFrameIntervals(True)
        win._refreshThreshold = 1/self.param_exp['framerate'] + 0.004 # i've got 50Hz monitor and want to allow 4ms tolerance

        # ---------------------------------------------------
        if verb: print('FPS = ',  win.getActualFrameRate() , 'framerate=', self.param_exp['framerate'])

        # ---------------------------------------------------
        target = visual.Circle(win, lineColor='white', size=self.param_exp['dot_size'], lineWidth=2)
        fixation = visual.GratingStim(win, mask='circle', sf=0, color='white', size=self.param_exp['dot_size'])
        ratingScale = visual.RatingScale(win, scale=None, low=-1, high=1, precision=100, size=.7, stretch=2.5,
                        labels=('Left', 'unsure', 'Right'), tickMarks=[-1, 0., 1], tickHeight=-1.0,
                        marker='triangle', markerColor='black', lineColor='White', showValue=False, singleClick=True,
                        showAccept=False, pos=(0, -self.param_exp['screen_height_px']/3)) #size=.4

        #Bip_pos = sound.Sound('2000', secs=0.05)
        #Bip_neg = sound.Sound('200', secs=0.5) # augmenter les fq

        # ---------------------------------------------------
        # fonction pause avec possibilité de quitter l'expérience
        msg_pause = visual.TextStim(win, text=u"\n\n\nTaper sur une touche pour continuer\n\nESCAPE pour arrêter l'expérience",
                                    font='calibri', height=25,
                                    alignHoriz='center')#, alignVert='top')

        text_score = visual.TextStim(win, font='calibri', height=30, pos=(0, self.param_exp['screen_height_px']/9))

        def pause(mode) :
            msg_pause.draw()
            win.flip()

            event.clearEvents()

            allKeys=event.waitKeys()
            for thisKey in allKeys:
                if thisKey in ['escape', 'a', 'q']:
                    core.quit()
                    win.close()
                    if mode=='eyeMvt' :
                        ET.End_trial()
                        ET.End_exp()
            if mode=='eyeMvt' :
                win.winHandle.set_fullscreen(False)
                win.winHandle.set_visible(False) # remis pour voir si ça enléve l'écran blanc juste après calibration
                ET.drift_correction()
                win.winHandle.set_visible(True) # remis pour voir si ça enléve l'écran blanc juste après calibration
                win.winHandle.set_fullscreen(True)

        def escape_possible(mode) :
            if event.getKeys(keyList=['escape', 'a', 'q']):
                win.close()
                core.quit()
                if mode=='eyeMvt' :
                    ET.End_trial()
                    ET.End_exp()

        # ---------------------------------------------------
        def presentStimulus_fixed(dir_bool):
            dir_sign = dir_bool * 2 - 1
            target.setPos((dir_sign * (self.param_exp['saccade_px']), self.param_exp['offset']))
            target.draw()
            win.flip()
            core.wait(0.3)

        clock = core.Clock()
        myMouse = event.Mouse(win=win)

        def presentStimulus_move(dir_bool):
            clock.reset()
            #myMouse.setVisible(0)
            dir_sign = dir_bool * 2 - 1
            while clock.getTime() < self.param_exp['stim_tau']:
                escape_possible(self.mode)
                # la cible à t=0 recule de sa vitesse * latency=RashBass (ici mis en s)
                target.setPos(((dir_sign * self.param_exp['V_X']*clock.getTime())-(dir_sign * self.param_exp['V_X']*(self.param_exp['RashBass']/1000)), self.param_exp['offset']))
                target.draw()
                win.flip()
                win.flip()
                escape_possible(self.mode)
                #win.flip()

        # ---------------------------------------------------
        # EXPERIMENT
        # ---------------------------------------------------
        if self.mode == 'bet' : results = np.zeros((self.param_exp['N_trials'], self.param_exp['N_blocks'] ))

        if self.mode == 'eyeMvt':
            ET.Start_exp()

            # Effectuez la configuration du suivi au début de l'expérience.
            win.winHandle.set_fullscreen(False)
            win.winHandle.set_visible(False) # remis pour voir si ça enléve l'écran blanc juste après calibration
            ET.calibration()
            win.winHandle.set_visible(True) # remis pour voir si ça enléve l'écran blanc juste après calibration
            win.winHandle.set_fullscreen(True)

        if self.mode == 'bet' : score = 0

        for block in range(self.param_exp['N_blocks']):

            x = 0

            if self.mode == 'bet' :
                text_score.text = '%1.0f/100' %(score / 50 * 100)
                text_score.draw()
                score = 0
            pause(self.mode)


            for trial in range(self.param_exp['N_trials']):

                # ---------------------------------------------------
                # PAUSE tous les 50 essais
                # ---------------------------------------------------
                if x == 50 :
                    if self.mode == 'bet' :
                        text_score.text = '%1.0f/100' %(score / 50 * 100)
                        text_score.draw()
                        score = 0

                    pause(self.mode)
                    x = 0

                x = x +1

                # ---------------------------------------------------
                # FIXATION
                # ---------------------------------------------------
                if self.mode == 'bet' :

                    event.clearEvents()
                    ratingScale.reset()

                    while ratingScale.noResponse :
                        fixation.draw()
                        ratingScale.draw()
                        escape_possible(self.mode)
                        win.flip()

                    ans = ratingScale.getRating()
                    results[trial, block] = ans

                if self.mode == 'eyeMvt':

                    ET.check()
                    ET.Start_trial(trial)

                    fixation.draw()
                    tps_start_fix = time.time()
                    win.flip()
                    escape_possible(self.mode)

                    ET.StimulusON(tps_start_fix)
                    ET.Fixation(fixation, tps_start_fix, win, escape_possible)

                # ---------------------------------------------------
                # GAP
                # ---------------------------------------------------
                win.flip()
                escape_possible(self.mode)
                if self.mode == 'eyeMvt': ET.StimulusOFF()
                core.wait(0.3)

                # ---------------------------------------------------
                # Mouvement cible
                # ---------------------------------------------------
                escape_possible(self.mode)
                dir_bool = self.param_exp['p'][trial, block, 0]
                if self.mode == 'eyeMvt': ET.TargetON()
                presentStimulus_move(dir_bool)
                escape_possible(self.mode)
                win.flip()

                if self.mode == 'bet' :
                    score_trial = ans * (dir_bool * 2 - 1)
                #    if score_trial > 0 :
                #        Bip_pos.setVolume(score_trial)
                #        Bip_pos.play()
                #    else :
                #        Bip_neg.setVolume(-score_trial)
                #        Bip_neg.play()
                #    core.wait(0.1)

                    score += score_trial

                if self.mode == 'eyeMvt':
                    ET.TargetOFF()
                    ret_value = ET.fin_enregistrement()
                    ET.check_trial(ret_value)

        if self.mode == 'bet' : self.param_exp['results'] = results

        if self.mode == 'eyeMvt': ET.End_exp()

        with open(self.exp_name(), 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(self.param_exp)

        win.update()
        core.wait(0.5)
        win.saveFrameIntervals(fileName=None, clear=True)

        win.close()

        core.quit()






#############################################################################
######################### ANALYSIS ##########################################
#############################################################################

def mutual_information(p, data, bin_p=20, bin_data=30):
    """
    Script to calculate Mutual Information between two discrete random variables
    Roberto maestre - rmaestre@gmail.com
    Bojan Mihaljevic - boki.mihaljevic@gmail.com

    Calculate and return Mutual information between two random variables
    https://github.com/rmaestre/Mutual-Information/blob/master/it_tool.py
    https://fr.wikipedia.org/wiki/Information_mutuelle
    """
    #----------------------------------------------------------------------
    #def mutual_information(hgram):
    #   """ Mutual information for joint histogram
    #   https://matthew-brett.github.io/teaching/mutual_information.html"""
    #   # Convert bins counts to probability values
    #   pxy = hgram / float(np.sum(hgram))
    #   px = np.sum(pxy, axis=1) # marginal for x over y
    #   py = np.sum(pxy, axis=0) # marginal for y over x
    #   px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    #   # Now we can do the calculation using the pxy, px_py 2D arrays
    #   nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    #   return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    #----------------------------------------------------------------------
    #x = np.round(p, decimals=base)
    ##y = np.round(data, decimals=base)
    #
    #if min(data)<0 : y = np.round(data, decimals=base-1)
    #else :           y = np.round(data, decimals=base)
    #
    #support_x = set(x)
    #support_y = set(y)

    #summation = 0.00
    #for value_x in support_x:
    #    for value_y in support_y:
    #        px = np.shape(np.where(x==value_x))[1] / len(x) #px=sum(x==value_x)/len(x) -> bcp plus long
    #        py = np.shape(np.where(y==value_y))[1] / len(x) #py=sum(y==value_y)/len(y) -> bcp plus long
    #        pxy = len(np.where(np.in1d(np.where(x==value_x)[0],
    #                                   np.where(y==value_y)[0])==True)[0]) / len(x)
    #        if pxy>0.00: summation += pxy * np.log2(pxy / (px*py))
    #----------------------------------------------------------------------

    x = p
    y = data

    support_x = np.linspace(min(x), max(x), bin_p)
    support_y = np.linspace(min(y), max(y), int(len(data)/bin_data))

    summation = 0.00
    for a in range(len(support_x)-1):
        for b in range(len(support_y)-1):
            ind_x = np.where((x>=support_x[a]) & (x<=support_x[a+1]))[0]
            ind_y = np.where((y>=support_y[b]) & (y<=support_y[b+1]))[0]

            px = len(ind_x) / len(x)
            py = len(ind_y) / len(x)
            pxy = len(np.where(np.in1d(ind_x, ind_y)==True)[0]) / len(x)

            if pxy>0.00: summation += pxy * np.log2(pxy / (px*py))


    return summation


def regress(ax, p, data, y1=0, y2=1, t_label=10, color='k', x1=-0.032, x2=1.032, pos='right', line=True, text=True, return_r_mi=False, lw=2) :
    from scipy import stats
    slope, intercept, r_, p_value, std_err = stats.linregress(p, data)
    x_test = np.linspace(np.min(p), np.max(p), 100)
    fitLine = slope * x_test + intercept

    #hist, x_edges, y_edges = np.histogram2d(p, data, bins=20)
    #hist, x_edges, y_edges = np.histogram2d(p, data, bins=len(data)/30)
    # mi =mutual_information(hist)
    mi = mutual_information(p, data)


    if line is True : ax.plot(x_test, fitLine, c=color, linewidth=lw)

    if pos[-5:]=='right' : x_pos=x2-(x2-x1)/10 ; h_pos='right'
    else:                  x_pos=x1+(x2-x1)/10 ; h_pos='left'
    if pos[:5]=='upper' : y_pos_r=y2-(y2-y1)/10 ; y_pos_m=y2-2*(y2-y1)/10
    else:                 y_pos_r=y1+(y2-y1)/10 ; y_pos_m=y1+2*(y2-y1)/10

    if text is True :
        ax.text(x_pos, y_pos_r, 'r = %0.3f'%(r_), color=color, fontsize=t_label/1.2, ha=h_pos)
        ax.text(x_pos, y_pos_m, 'MI = %0.3f'%(mi), color=color, fontsize=t_label/1.2, ha=h_pos)

    if return_r_mi is True : return ax, r_, mi
    else :                   return ax

def results_sujet(self, ax, sujet, s, mode_bcp, tau, t_label, pause, color = [['k', 'k'], ['r', 'r'], ['k','w']], alpha = [[.35,.15],[.35,.15],[1,0]], color_bcp='darkred'):

    import bayesianchangepoint as bcp
    from scipy import stats

    lw = 1.3
    ec = 0.2 # pour l'écart entre les différents blocks

    suj = sujet[s]

    print('Subject', suj, '=', self.subjects[suj])
    N_trials = self.BET[self.subjects[suj]]['N_trials']
    N_blocks = self.BET[self.subjects[suj]]['N_blocks']
    p = self.BET[self.subjects[suj]]['p']
    # tau = N_trials/5.
    h = 1./tau
    results = (self.BET[self.subjects[suj]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
    a_anti = self.EYEMVT[self.subjects[suj]]['a_anti']

    for block in range(N_blocks) :
        #----------------------------------------------------------------------------------
        ax.step(range(N_trials), block+p[:, block, 1]+ec*block, lw=1, alpha=alpha[1][0], c=color[1][0])
        ax.fill_between(range(N_trials), block+np.zeros_like(p[:, block, 1])+ec*block, block+p[:, block, 1]+ec*block,
                            lw=0, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
        ax.fill_between(range(N_trials), block+np.ones_like(p[:, block, 1])+ec*block, block+p[:, block, 1]+ec*block,
                            lw=0, alpha=alpha[1][1], facecolor=color[1][1], step='pre')

        ax.step(range(N_trials), block+((np.array(a_anti[block])-np.nanmin(a_anti))/(np.nanmax(a_anti)-np.nanmin(a_anti)))+ec*block,
                    color='k', lw=1.2, label='Eye movements' if block==0 else '')
        ax.step(range(N_trials), block+results[:, block]+ec*block, color='r', lw=1.2, label='Individual guess' if block==0 else '')
        #----------------------------------------------------------------------------------

        if mode_bcp is not None :
            if pause is True : liste = [0,50,100,150,200]
            else : liste = [0, 200]

            for a in range(len(liste)-1) :
                p_bar, r, beliefs = bcp.inference(p[liste[a]:liste[a+1], block, 0], h=h, p0=.5)
                p_hat, r_hat = bcp.readout(p_bar, r, beliefs,mode=mode_bcp)
                p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
                for i_trial in range(liste[a+1]-liste[a]):
                    p_low[i_trial], p_sup[i_trial] = stats.beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])

                ax.plot(np.arange(liste[a], liste[a+1]), block+p_hat+ec*block, c=color_bcp, alpha=.9, lw=1.5,
                        label='$\hat{x}_1$' if block==0 and a==0 else '')
                ax.plot(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block, c=color_bcp, ls='--', lw=1.2, label='CI' if block==0 and a==0 else '')
                ax.plot(np.arange(liste[a], liste[a+1]), block+p_low+ec*block, c=color_bcp, ls= '--', lw=1.2)
                ax.fill_between(np.arange(liste[a], liste[a+1]), block+p_sup+ec*block, block+p_low+ec*block, lw=.5, alpha=.11, facecolor=color_bcp)

    #------------------------------------------------
    # affiche les numéro des block sur le côté gauche
    #------------------------------------------------
    ax_block = ax.twinx()
    if s == 0 : ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
    ax_block.yaxis.set_label_coords(1.01, 1.08)
    ax_block.set_ylim(0, N_blocks)
    ax_block.set_yticks(np.arange(N_blocks)+0.5)
    ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.5)
    ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

    #------------------------------------------------
    ax.set_yticks([0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2])
    ax.set_yticklabels(['0','1','0','1','0','1'],fontsize=t_label/1.8)
    ax.yaxis.set_label_coords(-0.02, 0.5)
    ax.set_ylabel('Subject %s'%(suj), fontsize=t_label)
    ax.set_ylim(-(ec/2), N_blocks +ec*3-(ec/2))
    ax.set_xlim(0, N_trials)
    #------------------------------------------------

    #------------------------------------------------
    # Barre Pause
    #------------------------------------------------
    ax.bar(50, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
    ax.bar(100, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)
    ax.bar(150, 3+ec*3, bottom=-ec/2, color='k', width=.2, linewidth=0)

    return ax


class Analysis(object):
    """ docstring for the aSPEM class. """

    def __init__(self, observer=None, mode=None, name_file_fit='fct_velocity_sigmo') :
        self.subjects = ['s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12']
        self.name_file_fit = name_file_fit
        self.mode = mode
        self.observer = observer
        self.init()

    def init(self) :

        self.dry_run = True
        self.dry_run = False

        # ---------------------------------------------------
        # setup values
        # ---------------------------------------------------
        cachedir = 'data_cache'
        datadir = 'data'
        import os
        for dir_ in [datadir, cachedir] :
            try: os.mkdir(dir_)
            except: pass

        # ---------------------------------------------------
        # récuperation de toutes les données
        # ---------------------------------------------------
        import glob
        liste = {'bet':{}, 'eyeMvt':{}}
        for fname in glob.glob('data/*pkl'):
            a = fname.split('/')[1].split('.')[0].split('_')
            liste[a[0]][a[1]] = a[2]+'_'+a[3]

        self.BET = {}
        for s in liste['bet'].keys() :
            if s in self.subjects :
                a = 'data/bet_%s_%s.pkl'%(s, liste['bet'][s])
                with open(a, 'rb') as fichier :
                    b = pickle.load(fichier, encoding='latin1')
                    self.BET[s] = b

        a = 'parametre/Delete/Delete_list_BadTrials_Full_%s.pkl'%(self.name_file_fit)
        try :
            with open(a, 'rb') as fichier :
                b = pickle.load(fichier, encoding='latin1')
                self.list_delete = b
        except :
            self.list_delete = None
            print('/!\ Le fichier Delete n\'existe pas pour %s !'%(self.name_file_fit))

        self.EYEMVT = {}
        for s in liste['eyeMvt'].keys() :
            if s in self.subjects:
                a = 'parametre/%s/param_Fit_%s_%s.pkl'%(self.name_file_fit, s, self.name_file_fit)
                try :
                    with open(a, 'rb') as fichier :
                        b = pickle.load(fichier, encoding='latin1')
                        self.EYEMVT[s] = b
                except :
                    print('/!\ Le fichier param Fit n\'existe pas pour %s !'%(s))

        # ---------------------------------------------------
        if self.observer is None : self.observer =  's08'
        if self.mode is None : self.mode = 'eyeMvt'

        self.timeStr = liste[self.mode][self.observer]

        self.param_exp = self.BET[self.observer]
        if self.mode == 'eyeMvt' : self.param = self.EYEMVT[self.observer]



    def Full_list(self, modes_bcp=['expectation', 'max', 'mean', 'fixed', 'leaky', 'hindsight'], pause=True):

        import pandas as pd
        pd.set_option('mode.chained_assignment', None)

        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']

        p = self.param_exp['p']

        full = pd.DataFrame(index=np.arange(len(self.subjects)*N_trials*N_blocks), columns=('sujet', 'proba','bino','results','aa','va'))

        if modes_bcp is not None :

            if type(modes_bcp) is not list : modes_bcp = [modes_bcp]

            import bayesianchangepoint as bcp
            for m in modes_bcp : full['p_hat_%s'%m] = np.arange(len(self.subjects)*N_trials*N_blocks)*np.nan

        for i, suj in enumerate(self.subjects) :

            results = (self.BET[suj]['results']+1)/2
            a_anti = self.EYEMVT[suj]['a_anti']
            start_anti = self.EYEMVT[suj]['start_anti']
            latency = self.EYEMVT[suj]['latency']

            for block in range(N_blocks):

                nb = i*N_trials*N_blocks
                a = nb + N_trials*block
                b = (nb + N_trials*(block+1))

                full['sujet'][a:b] = suj
                full['proba'][a:b] = p[:, block, 1]
                full['bino'][a:b] = p[:, block, 0]
                full['results'][a:b] = results[:, block]
                full['aa'][a:b] = a_anti[block]
                full['va'][a:b] = (np.array(a_anti[block])*((np.array(latency[block])-np.array(start_anti[block]))/1000))

                for t in self.list_delete[suj][block] :
                    full['aa'][a+t] = np.nan
                    full['va'][a+t] = np.nan

                if modes_bcp is not None :
                    tau = N_trials/5.
                    h = 1./tau

                    p_hat_block = {}

                    for m in modes_bcp : p_hat_block[m] = []

                    if pause is True :
                        liste = [0,50,100,150,200]
                        for s in range(len(liste)-1) :
                            p_bar, r_bar, beliefs = bcp.inference(p[liste[s]:liste[s+1], block, 0], h=h, p0=.5)

                            for m in modes_bcp :
                                p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=m)
                                p_hat_block[m].extend(p_hat)

                    else :
                        p_bar, r_bar, beliefs = bcp.inference(p[:, block, 0], h=h, p0=.5)
                        for m in modes_bcp :
                            p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=m)
                            p_hat_block[m] = p_hat

                    for m in modes_bcp : full['p_hat_%s'%m][a:b] = p_hat_block[m]

        return full

    def Data_Scalling(self):

        Full = Analysis.Full_list(self, modes_bcp=None)
        scal_va_sujet, scal_va_full = {}, []
        scal_bet_sujet, scal_bet_full = {}, []

        va_full, bet_full = [], []

        for x in self.subjects :
            bet = list(Full['results'][Full.sujet==x])
            va = list(Full['va'][Full.sujet==x])

            bet_full.extend(bet)
            va_full.extend(va)

            scal_bet_sujet[x] = np.sort(bet)
            scal_va_sujet[x] = np.sort(va)

        scal_va_full = np.sort(va_full)
        scal_bet_full = np.sort(bet_full)

        new_proba_full = np.linspace(0,1,len(scal_va_full))

        new_va_sujet, new_va_full = {}, {}
        new_bet_sujet, new_bet_full = {}, {}


        for x in self.subjects :

            N_trials = self.param_exp['N_trials']
            N_blocks = self.param_exp['N_blocks']

            new_proba_sujet = np.linspace(0, 1, len(scal_va_sujet[x]))
            va_sujet = list(Full['va'][Full.sujet==x])
            bet_sujet = list(Full['results'][Full.sujet==x])

            nb_trial = 0

            new_va_sujet[x], new_va_full[x] = [], []
            new_bet_sujet[x], new_bet_full[x] = [], []

            for block in range(N_blocks) :
                new_va_sujet[x].append([])
                new_va_full[x].append([])

                new_bet_sujet[x].append([])
                new_bet_full[x].append([])

                va = va_sujet[nb_trial : N_trials + nb_trial*block]
                bet = bet_sujet[nb_trial : N_trials + nb_trial*block]

                for trial in range(N_trials) :
                    new_va_sujet[x][block].append(np.mean([new_proba_sujet[t] for t in range(len(scal_va_sujet[x]))
                                                           if scal_va_sujet[x][t]==va[trial]]))

                    new_va_full[x][block].append(np.mean([new_proba_full[t] for t in range(len(scal_va_full))
                                                          if scal_va_full[t]==va[trial]]))

                    new_bet_sujet[x][block].append(np.mean([new_proba_sujet[t] for t in range(len(scal_bet_sujet[x]))
                                                            if scal_bet_sujet[x][t]==bet[trial]]))

                    new_bet_full[x][block].append(np.mean([new_proba_full[t] for t in range(len(scal_bet_full))
                                                           if scal_bet_full[t]==bet[trial]]))

                nb_trial = nb_trial + N_trials

        new_data = {}
        new_data['new_bet_full'] = new_bet_full
        new_data['new_bet_sujet'] = new_bet_sujet
        new_data['new_va_full'] = new_va_full
        new_data['new_va_sujet'] = new_va_sujet

        return new_data

    def Find_h(self, new_bet, new_va, modes_bcp='mean') :

        from lmfit import  Model, Parameters
        from lmfit import minimize
        import bayesianchangepoint as bcp

        def fct_BCP(x, tau, fixed_window, sujet, block, cent) :
            # TODO merge des blocks puis  range(0, 600, 50) pour les pauses plus param = taille de window
            h = 1/tau
            p_hat = np.zeros(len(x))

            if sujet==True :
                for b in range(self.param_exp['N_blocks']):
                    nb = self.param_exp['N_trials']*b
                    liste = [0, 50, 100, 150, 200]
                    for a in range(len(liste)-1) :
                        p_bar, r_bar, beliefs = bcp.inference(x[nb+liste[a]:nb+liste[a+1]], h=h, p0=.5, r0=1.)
                        p_hat_p, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=modes_bcp, p0=.5, fixed_window_size=fixed_window)
                        p_hat[nb+liste[a]:nb+liste[a+1]] = p_hat_p

            elif block==True :
                liste = [0, 50, 100, 150, 200]
                for a in range(len(liste)-1) :
                    p_bar, r_bar, beliefs = bcp.inference(x[liste[a]:liste[a+1]], h=h, p0=.5, r0=1.)
                    p_hat_p, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=modes_bcp, p0=.5, fixed_window_size=fixed_window)
                    p_hat[liste[a]:liste[a+1]] = p_hat_p

            elif cent==True :
                liste = [0, 50, 100]
                for a in range(len(liste)-1) :
                    p_bar, r_bar, beliefs = bcp.inference(x[liste[a]:liste[a+1]], h=h, p0=.5, r0=1.)
                    p_hat_p, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=modes_bcp, p0=.5, fixed_window_size=fixed_window)
                    p_hat[liste[a]:liste[a+1]] = p_hat_p

            else :
                p_bar, r_bar, beliefs = bcp.inference(x, h=h, p0=.5, r0=1.)
                p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=modes_bcp, p0=.5, fixed_window_size=fixed_window)

            return p_hat

        def KL_distance(p_data, p_hat):
            distance = p_hat * np.log2(p_hat) - p_hat * np.log2(p_data + 1.*(p_data==0.))
            distance += (1-p_hat) * np.log2(1-p_hat) - (1-p_hat) * np.log2(1-p_data + 1.*(p_data==1.))
            return distance

        def residual(params, x, data):
            model = fct_BCP(x, params['tau'], params['fixed_window'], params['sujet'], params['block'], params['cent'])
            return KL_distance(data, model)

        def fit(h, x, bet, va, sujet=False, block=False, cent=False):

            tau = 1/h
            x, bet, va = np.array(x), np.array(bet), np.array(va)

            params = Parameters()
            if modes_bcp in ['leaky', 'fixed'] :
                params.add('tau', value=tau, vary=False)
                params.add('fixed_window', value=tau, min=1)
            else :
                params.add('tau', value=tau, min=1)
                params.add('fixed_window', value=tau, vary=False)

            params.add('sujet', value=sujet, vary=False)
            params.add('block', value=block, vary=False)
            params.add('cent', value=cent, vary=False)

            result_res =   minimize(residual, params, args=(x, bet), nan_policy='omit')
            result_v_ant = minimize(residual, params, args=(x, va), nan_policy='omit')

            if modes_bcp in ['leaky', 'fixed'] :
                h_bet = 1/result_res.params['fixed_window'].value
                h_va =  1/result_v_ant.params['fixed_window'].value
            else :
                h_bet = 1/result_res.params['tau'].value
                h_va =  1/result_v_ant.params['tau'].value

            return h_bet, h_va

        h_bet, h_va = {}, {}
        for l in ['pause', 'block', 'sujet', '100'] : h_bet[l], h_va[l] = {}, {}

        for x, sujet in enumerate(self.subjects) :

            print(sujet, end=' ')

            prob_sujet, bet_sujet, a_anti_sujet = [], [], []

            p = p = self.param_exp['p']
            tau = self.param_exp['N_trials']/5.
            h = 1./tau

            for l in ['pause', 'block', 'sujet', '100'] : h_bet[l][sujet], h_va[l][sujet] = [], []

            #----------------------------------------------------
            # BLOCK
            #----------------------------------------------------
            for block in range(self.param_exp['N_blocks']):

                bet = new_bet[sujet][block]
                va = new_va[sujet][block]

                prob_block = p[:, block, 0]
                h_b, h_v = fit(h, prob_block, bet, va, block=True)

                h_bet['block'][sujet].append(h_b)
                h_va['block'][sujet].append(h_v)

                prob_sujet.extend(p[:, block, 0])
                bet_sujet.extend(bet)
                a_anti_sujet.extend(va)

                #----------------------------------------------------
                # PAUSE
                #----------------------------------------------------
                liste = [0,50,100,150,200]
                for a in range(len(liste)-1) :
                    va_p = va[liste[a]:liste[a+1]]
                    bet_p = bet[liste[a]:liste[a+1]]
                    prob_pause = p[liste[a]:liste[a+1], block, 0]

                    h_b, h_v = fit(h, prob_pause, bet_p, va_p)

                    h_bet['pause'][sujet].append(h_b)
                    h_va['pause'][sujet].append(h_v)

            #----------------------------------------------------
            # SUJET
            #----------------------------------------------------
            h_b, h_v = fit(h, prob_sujet, bet_sujet, a_anti_sujet, sujet=True)
            h_bet['sujet'][sujet].append(h_b)
            h_va['sujet'][sujet].append(h_v)


            #----------------------------------------------------
            # 100 Trials
            #----------------------------------------------------
            for a in range(0, self.param_exp['N_blocks']*self.param_exp['N_trials']-50, 50) :
                h_b, h_v = fit(h, prob_sujet[a:a+100], bet_sujet[a:a+100], a_anti_sujet[a:a+100], cent=True)
                h_bet['100'][sujet].append(h_b)
                h_va['100'][sujet].append(h_v)

        return h_bet, h_va





    #------------------------------------------------------------------------------------------------------------------------

    def plot_equation(self, equation='fct_velocity', fig_width=15, t_titre=35, t_label=20) :

        '''
        Returns figure of the equation used for the fit with the parameters of the fit

        Parameters
        ----------
        equation : str or function
            if 'fct_velocity' displays the fct_velocity equation
            if 'fct_position' displays the fct_position equation
            if 'fct_saccades' displays the fct_saccades equation
            if function displays the function equation

        fig_width : int
            figure size

        t_titre : int
            size of the title of the figure

        t_label : int
            size x and y label

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure
        ax : AxesSubplot
            figure
        '''

        # from pygazeanalyser.edfreader import read_edf
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO
        Plot = ANEMO.Plot(self.param_exp)

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        fig, axs = Plot.plot_equation(equation=equation, fig_width=fig_width, t_titre=t_titre, t_label=t_label)

        return fig, axs

    def plot_data(self, show='velocity', trials=0, block=0,
                    N_trials=None,
                    fig_width=15, t_titre=35, t_label=20,
                    stop_search_misac=None, name_trial_show=False, before_sacc=5, after_sacc=15) :
        '''
        Returns the data figure

        Parameters
        ----------
        show : str
            if 'velocity' show the velocity of the eye
            if 'position' show the position of the eye
            if 'saccades' shows the saccades of the eye

        trials : int or list
            number or list of trials to display
        block : int
            number of the block in which it finds the trials to display
        N_trials : int
            number of trials per block
            if None went searched in param_exp

        before_sacc: int
            time to remove before saccades
                it is advisable to put :
                    5 for 'fct_velocity' and 'fct_position'
                    0 for 'fct_saccade'

        after_sacc: int
            time to delete after saccades
                it is advisable to put : 15

        stop_search_misac : int
            stop search of micro_saccade
            if None: stops searching at the end of fixation + 100ms
        name_trial_show : bool
            if True the num is written of the trial in y_label

        fig_width : int
            figure size
        t_titre : int
            size of the title of the figure
        t_label : int
            size x and y label

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure
        ax : AxesSubplot
        figure
        '''

        # from pygazeanalyser.edfreader import read_edf
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO
        Plot = ANEMO.Plot(self.param_exp)

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        fig, axs = Plot.plot_data(data, show=show, trials=trials, block=block,
                                    N_trials=N_trials,
                                    fig_width=fig_width, t_titre=t_titre, t_label=t_label,
                                    stop_search_misac=stop_search_misac, show_num_trial=name_trial_show, before_sacc=before_sacc, after_sacc=after_sacc)

        return fig, axs


    def plot_Full_data(self, show='velocity', N_blocks=None,
                        N_trials=None,
                        fig_width=12, t_titre=20, t_label=14,
                        stop_search_misac=None, file_fig=None) :

        '''
        Save the full data figure

        Parameters
        ----------
        show : str
            if 'velocity' show velocity of the eye
            if 'position' show the position of the eye
            if 'saccades' shows the saccades of the eye

        N_blocks : int
            number of blocks
            if None went searched in param_exp
        N_trials : int
            number of trials per block
            if None went searched in param_exp

        stop_search_misac : int
            stop search of micro_saccade
            if None: stops searching at the end of fixation + 100ms

        fig_width : int
            figure size
        t_titre : int
            size of the title of the figure
        t_label : int
            size x and y label

        file_fig : str
            name of file figure reccorded
            if None file_fig is show

        Returns
        -------
        save the figure
        '''

        # from pygazeanalyser.edfreader import read_edf
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO
        Plot = ANEMO.Plot(self.param_exp)

        resultats = os.path.join(self.param_exp['datadir'], self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        if file_fig is None : file_fig = 'figures/%s_%s'%(show, self.observer)

        Plot.plot_Full_data(data, show=show, N_blocks=N_blocks,
                            N_trials=N_trials,
                            fig_width=fig_width, t_titre=t_titre, t_label=t_label,
                            stop_search_misac=stop_search_misac, file_fig=file_fig)

    def plot_fit(self, equation='fct_velocity', trials=0, block=0, N_trials=None,
                        fig_width=15, t_titre=35, t_label=20,
                        report=None, before_sacc=5, after_sacc=15,
                        step_fit=2, do_whitening=False, time_sup=280, param_fit=None, inde_vars=None,
                        stop_search_misac=None) :
        '''
        Returns figure of data fits

        Parameters
        ----------

        equation : str or function
            if 'fct_velocity' : does a data fit with the function 'fct_velocity'
            if 'fct_position' : does a data fit with the function 'fct_position'
            if 'fct_saccades' : does a data fit with the function 'fct_saccades'
            if function : does a data fit with the function

        trials : int or list
            number or list of trials to display
        block : int
            number of the block in which it finds the trials to display
        N_trials : int
            number of trials per block
            if None went searched in param_exp

        stop_search_misac : int
            stop search of micro_saccade
            if None: stops searching at the end of fixation + 100ms


        report : bool
            if true return the report of the fit for each trial
        step_fit : int
            number of steps for the fit
        do_whitening : bool
            if true the fit perform on filtered data with a whitening filter

        time_sup: int
            time that will be deleted to perform the fit (for data that is less good at the end of the test)
        param_fit : dict
            dictionary containing the parameters of the fit
        inde_vars : dict
            dictionary containing the independent variables of the fit

        before_sacc: int
            time to remove before saccades
                it is advisable to put :
                    5 for 'fct_velocity' and 'fct_position'
                    0 for 'fct_saccade'

        after_sacc: int
            time to delete after saccades
                it is advisable to put : 15


        fig_width : int
            figure size
        t_titre : int
            size of the title of the figure
        t_label : int
            size x and y label


        Returns
        -------
        fig : matplotlib.figure.Figure
            figure
        ax : AxesSubplot
            figure
        report : list
            list of the reports of the fit for each trial
        '''

        # from pygazeanalyser.edfreader import read_edf
        from ANEMO import read_edf
        #from edfreader import read_edf
        from ANEMO import ANEMO
        Plot = ANEMO.Plot(self.param_exp)

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        if report is None :
            fig, axs = Plot.plot_fit(data, equation=equation, trials=trials, block=block, N_trials=N_trials,
                                        fig_width=fig_width, t_titre=t_titre, t_label=t_label,
                                        report=report, before_sacc=before_sacc, after_sacc=after_sacc,
                                        step_fit=step_fit, do_whitening=do_whitening, time_sup=time_sup, param_fit=param_fit, inde_vars=inde_vars,
                                        stop_search_misac=stop_search_misac)

            return fig, axs

        else :
            fig, axs, results = Plot.plot_fit(data, equation=equation, trials=trials, block=block, N_trials=N_trials,
                                                fig_width=fig_width, t_titre=t_titre, t_label=t_label,
                                                report=report, before_sacc=before_sacc, after_sacc=after_sacc,
                                                step_fit=step_fit, do_whitening=do_whitening, time_sup=time_sup, param_fit=param_fit, inde_vars=inde_vars,
                                                stop_search_misac=stop_search_misac)

            return fig, axs, results


    def Fit (self, equation='fct_velocity', fitted_data='velocity',
                N_blocks=None, N_trials=None, list_param_enre=None,
                plot=True, file_fig=None, file_save=None,
                param_fit=None, inde_vars=None, step_fit=2,
                do_whitening=False, time_sup=280, before_sacc=5, after_sacc=15,
                stop_search_misac=None,
                fig_width=12, t_label=20, t_text=14) :
        '''
        Return the parameters of the fit present in list_param_enre

        Parameters
        ----------
        data : list
            edf data for the trials recorded by the eyetracker transformed by the read_edf function of the edfreader module

        equation : str or function
            if 'fct_velocity' : does a data fit with the function 'fct_velocity'
            if 'fct_position' : does a data fit with the function 'fct_position'
            if 'fct_saccades' : does a data fit with the function 'fct_saccades'
            if function : does a data fit with the function



        fitted_data : bool
            if 'velocity' = fit the velocity data for a trial in deg/sec
            if 'position' = fit the position data for a trial in deg
            if 'saccade' = fit the position data for sacades in trial in deg

        N_blocks : int
            number of blocks
            if None went searched in param_exp
        N_trials : int
            number of trials per block
            if None went searched in param_exp

        list_param_enre : list
            list of fit parameters to record
            if None :
                if equation in ['fct_velocity', 'fct_position'] : ['fit', 'start_anti', 'a_anti', 'latency', 'tau', 'maxi', 'saccades', 'old_anti', 'old_max', 'old_latency']
                if equation is 'fct_saccades' : ['fit', 'T0', 't1', 't2', 'tr', 'x_0', 'x1', 'x2', 'tau']

        plot : bool
            if true : save the figure in file_fig
        file_fig : str
            name of file figure reccorded
            if None file_fig is 'figures/Fit_%s_%s_%s_step_%s_whitening'%(self.observer, equation, step_fit, do_whitening)
        file_save : str
            name of file param reccorded
            if None file_fig is 'param_Fit_%s_%s_%s_step_%s_whitening.pkl'%(self.observer, equation, step_fit, do_whitening)

        param_fit : dic
            fit parameter dictionary, each parameter is a dict containing :
                'name': name of the variable,
                'value': initial value,
                'min': minimum value,
                'max': maximum value,
                'vary': True if varies during fit, 'vary' if only varies for step 2, False if not varies during fit
            if None : Generate by generation_param_fit
        inde_vars : dic
            independent variable dictionary of fit
            if None : Generate by generation_param_fit

        step_fit : int
            number of steps for the fit
        do_whitening : bool
            if True return the whitened fit
        time_sup: int
            time that will be deleted to perform the fit (for data that is less good at the end of the test)

        before_sacc: int
            time to remove before saccades
                it is advisable to put :
                    5 for 'fct_velocity' and 'fct_position'
                    0 for 'fct_saccade'

        after_sacc: int
            time to delete after saccades
                it is advisable to put : 15

        stop_search_misac : int
            stop search of micro_saccade
            if None: stops searching at the end of fixation + 100ms


        fig_width : int
            figure size
        t_label : int
            size x and y label
        t_text : int
            size of the text of the figure

        Returns
        -------
        param : dict
            each parameter are ordered : [block][trial]
        '''
        import matplotlib.pyplot as plt
        from ANEMO import read_edf
        from ANEMO import ANEMO
        Fit = ANEMO.Fit(self.param_exp)

        resultats = os.path.join('data', self.mode + '_' + self.observer + '_' + self.timeStr + '.asc')
        data = read_edf(resultats, 'TRIALID')

        if plot is True :
            if file_fig is None :
                if not os.path.exists('figures/Fit_%s'%equation): os.makedirs('figures/Fit_%s'%equation)
                file_fig='figures/Fit_%s/Fit_%s_%s_%s_step_%s_whitening'%(equation, self.observer, equation, step_fit, do_whitening)

        param = Fit.Fit_full(data, equation=equation, fitted_data=fitted_data,
                                N_blocks=N_blocks, N_trials=N_trials, list_param_enre=list_param_enre,
                                plot=plot, file_fig=file_fig,
                                param_fit=param_fit, inde_vars=inde_vars, step_fit=step_fit,
                                do_whitening=do_whitening, time_sup=time_sup, before_sacc=before_sacc, after_sacc=after_sacc,
                                stop_search_misac=stop_search_misac,
                                fig_width=fig_width, t_label=t_label, t_text=t_text)

        if file_save is None :
            if not os.path.exists('parametre'):
                os.makedirs('parametre')
            name_param = 'param_Fit_%s_%s_%s_step_%s_whitening.pkl'%(self.observer, equation, step_fit, do_whitening)
            file = os.path.join('parametre', name_param)
        else :
            file = file_save

        with open(file, 'wb') as fichier:
            f = pickle.Pickler(fichier)
            f.dump(param)

        print('END !!!')

    #------------------------------------------------------------------------------------------------------------------------


    def Plot_Average_Trace_P_real(self, delta, color, mean=False, pas_tps=10, ax=None, stop=610, show='r+l', title='', fig_width=15, t_titre=0, t_label=30*3) :

        from ANEMO import ANEMO
        from ANEMO import read_edf
        A = ANEMO(self.param_exp)
        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']
        p = self.param_exp['p']

        import glob
        timeStr = {}
        for fname in glob.glob('data/*pkl'):
            a = fname.split('/')[1].split('.')[0].split('_')
            if a[1] in self.subjects and a[0] == self.mode : timeStr[a[1]] = a[2]+'_'+a[3]



        if ax is None :
            fig_width = 20
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, 1*fig_width/(1.6180*1)))
        ax.plot(np.zeros(stop), c='k', alpha=0.6)

        v_r, v_l = {}, {}
        PROBA = np.arange(0,1,delta)
        for p_r in PROBA :
            v_r[p_r], v_l[p_r] = [], []

        if mean is not True : pas_tps = 1

        for x in range(len(self.subjects)) :

            resultats = os.path.join('data', self.mode + '_' + self.subjects[x] + '_' + timeStr[self.subjects[x]] + '.asc')
            data = read_edf(resultats, 'TRIALID')

            for block in range(N_blocks) :
                for trial in range(N_trials) :
                    p_reel = p[trial, block, 1]
                    trial_data = trial + N_trials*block

                    arg = A.arg(data[trial_data], trial=trial, block=block)
                    start = arg.TargetOn-arg.t_0-300

                    for p_r in PROBA :
                        if p_reel >= p_r and p_reel < (p_r + delta) :
                            if arg.dir_target == (-1) : # droite c'est 1 gauche c'est -1
                                if show in ['l', 'r+l'] :
                                    velocity_NAN = A.velocity_NAN(arg.data_x, arg.data_y, arg.saccades,
                                                      arg.trackertime, arg.TargetOn,
                                                      before_sacc=5, after_sacc=15)
                                    v_l[p_r].append(velocity_NAN[start:start+stop])

                            elif arg.dir_target == 1 :
                                if show in ['r', 'r+l'] :
                                    velocity_NAN = A.velocity_NAN(arg.data_x, arg.data_y, arg.saccades,
                                                      arg.trackertime, arg.TargetOn,
                                                      before_sacc=5, after_sacc=15)
                                    v_r[p_r].append(velocity_NAN[start:start+stop])

        x=0
        for p_r in PROBA :
            mean_v_r, mean_v_l = [], []
            std_v_r, std_v_l = [], []

            for tps in range(stop) :
                if show in ['r', 'r+l'] :
                    liste_r = []
                    for a in range(len(v_r[p_r])) : liste_r.append(v_r[p_r][a][tps])
                    mean_v_r.append(np.nanmean(liste_r))
                    std_v_r.append(np.nanstd(liste_r))

                if show in ['l', 'r+l'] :
                    liste_l = []
                    for b in range(len(v_l[p_r])) : liste_l.append(v_l[p_r][b][tps])
                    mean_v_l.append(np.nanmean(liste_l))
                    std_v_l.append(np.nanstd(liste_l))

            if mean is True :
                mean_m_r, mean_m_l = [], []
                std_m_r, std_m_l = [], []

                if show == 'r' : len_mean = len(mean_v_r)
                else : len_mean = len(mean_v_l)

                for t in np.arange(0,len_mean ,pas_tps) :
                    if show in ['r', 'r+l'] :
                        mean_m_r.append(np.nanmean(mean_v_r[t:t+pas_tps]))
                        std_m_r.append(np.nanmean(std_v_r[t:t+pas_tps]))
                    if show in ['l', 'r+l'] :
                        std_m_l.append(np.nanmean(std_v_l[t:t+pas_tps]))
                        mean_m_l.append(np.nanmean(mean_v_l[t:t+pas_tps]))

            else :
                if show in ['r', 'r+l'] : mean_m_r, std_m_r  = mean_v_r, std_v_r
                if show in ['l', 'r+l'] : mean_m_l, std_m_l = mean_v_l, std_v_l

            if show in ['r', 'r+l'] :
                mean_r, std_r = np.asarray(mean_m_r), np.asarray(std_m_r)
                ax.plot(mean_r, c=color[x], lw=3, alpha=1, label=' p = %.1f - %.1f'%(p_r, p_r+delta))
                ax.fill_between(range(int(stop/pas_tps)), mean_r+std_r, mean_r-std_r, facecolor=color[x], alpha=0.05)

            if show in ['l', 'r+l'] :
                if show == 'l' : label_l = ' p = %.1f - %.1f'%(p_r, p_r+delta)
                else : label_l = None
                mean_l, std_l = np.asarray(mean_m_l), np.asarray(std_m_l)
                ax.plot(mean_l, c=color[x], lw=2.5, alpha=1, label=label_l)
                ax.fill_between(range(int(stop/pas_tps)), mean_l+std_l, mean_l-std_l, facecolor=color[x], alpha=0.05)

            x=x+1

        if show == 'r' : min_y, max_y = -11.28, 21.28
        if show == 'l' : min_y, max_y =  -21.28, 11.28
        if show == 'r+l' : min_y, max_y = -21.28, 21.28

        ax.axis([0, (stop/pas_tps)-pas_tps, min_y, max_y])

        ax.axvspan(0, int(300/pas_tps), color='r', alpha=0.2)
        ax.axvspan(int(300/pas_tps), int(stop/pas_tps), color='k', alpha=0.15)

        # COSMETIQUE
        ax.text(int(300/pas_tps)/2, min_y+(max_y-min_y)/10, "GAP", color='k', fontsize=t_label/1., ha='center', va='center', alpha=0.5)
        ax.text(int(300/pas_tps)+(int((stop-300)/pas_tps))/2, min_y+(max_y-min_y)/10, "PURSUIT", color='k', fontsize=t_label/1., ha='center', va='center', alpha=0.5)

        ax.legend(loc=2, fontsize=t_label/1.8, framealpha=0.3)
        ax.set_title(title, fontsize=t_titre/1.2)
        ax.set_xlabel('Time (ms)', fontsize=t_label/1.2)
        ax.set_ylabel('Velocity (°/s)', fontsize=t_label/1.2)

        ax.tick_params(axis='both', labelsize=t_label/1.8)
        ax.set_xticks(np.arange(0,(stop/pas_tps)+1,pas_tps))
        ax.set_xticklabels(np.arange(-300,stop-300+1,100))

        return ax


    def plot_bcp(self, show_trial=False, block=0, trial=50, N_scan=100, fixed_window_size=40, label_bcp=r'$P_{BBCP}$', label_comp_bcp=r'$P_{leaky}$',
                pause=None, mode=['expectation', 'max', 'mean', 'fixed', 'leaky', 'hindsight'],
                mode_compare=None, max_run_length=150, c_mode='g', c_compare='r', TD=False, lw=2,
                color=[['k', 'k'], ['r', 'r'], ['k','w']], alpha = [[.35,.15],[.35,.15],[1,0]],
                fig_width=15, t_titre=35, t_label=20, show_title=True, leg_up=None):

        '''plot='normal' -> bcp, 'detail' -> bcp2'''

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import bayesianchangepoint as bcp
        from scipy.stats import beta

        if type(mode) is not list : mode = [mode]

        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']
        p = self.param_exp['p']
        tau = N_trials/5.
        h = 1/tau

        p0, r0 =  0.5, 1.0

        def plot_result_bcp(ax1, ax2, mode, observation, time, c, label, name=True, show_run_lengths=True, show_r_hat=True) :

            p_bar, r_bar, beliefs = bcp.inference(observation, h=h, p0=p0, r0=r0)
            p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=mode, fixed_window_size=fixed_window_size, p0=p0)
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)
            N_r, N_trial = beliefs.shape

            for i_trial in range(N_trial):
                p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
            ax1.plot(time, p_hat, c=c,  lw=lw, alpha=.9, label=label if name is True else '')
            ax1.plot(time, p_sup, c=c, lw=lw, alpha=.9, ls='--')#, label='CI '+label if name is True else '')
            ax1.plot(time, p_low, c=c, lw=lw, alpha=.9, ls='--')
            ax1.fill_between(time, p_sup, p_low, lw=.5, alpha=.11, facecolor=c)

            if ax2 is not None :

                if show_run_lengths is True :
                    if N_trial < N_trials : extent = (min(time), max(time), np.max(r_bar), np.min(r_bar))
                    else : extent = None

                    eps=1.e-5 # 1.e-12
                    #ax2.imshow(np.log(beliefs[:max_run_length, :] + eps), cmap='Greys', extent=extent)
                    if mode == 'fixed':
                        ax2.imshow(np.log(beliefs[:max_run_length, :]*0. + eps), cmap='Greys', extent=extent)
                    elif mode == 'leaky':
                        beliefs_ = np.exp(-np.arange(N_r) / fixed_window_size)
                        beliefs_ /= beliefs_.sum()
                        beliefs_ = beliefs_[:, None]
                        ax2.imshow(np.log((beliefs_*np.ones(N_trial))[:max_run_length, :] + eps), cmap='Greys', extent=extent)
                    else:
                        ax2.imshow(np.log(beliefs[:max_run_length, :] + eps), cmap='Greys', extent=extent)
                    ax2.set_ylim(0, max_run_length)
                if show_r_hat is True :
                    ax2.plot(time, r_hat, c=c, lw=lw, alpha=1, label='predicted run-length')


                return (ax1, ax2)

            else :
                return ax1


        height_ratios = np.ones(len(mode))


        if show_trial is True :
            print('Block', block)
            height_ratios = np.append(height_ratios, 1/4)
            nb_fig = len(mode)+1
            figsize=(fig_width, (nb_fig)*(fig_width)/(2*(1.6180)))

        else:
            nb_fig = len(mode)


        if N_scan>0: #show_trial is False :

            #---------------------------------------------------------------------------
            # SCORE
            #---------------------------------------------------------------------------
            border = 2*int(tau)
            hs = h*np.logspace(-2, 1, N_scan)
            score = np.zeros((len(mode), N_scan, N_blocks))
            #KL = np.zeros((len(modes), N_scan, N_blocks))
            figsize=(fig_width, nb_fig*(fig_width)/(2*(1.6180)))

            for i_block in range(N_blocks):
                o = p[:, i_block, 0]
                for i_scan, h_ in enumerate(hs):
                    p_bar, r_bar, beliefs = bcp.inference(o, h=h_, p0=p0, r0=r0)
                    for i_mode, m in enumerate(mode):
                        if m=='fixed': p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=m, fixed_window_size=int(1/h_))
                        else: p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=m, p0=p0)

                        score[i_mode, i_scan, i_block] = np.mean(np.log2(bcp.likelihood(o[(border+1):], p_hat[border:-1], r_hat[border:-1])))
                        #KL_ = p_hat * np.log2(p_hat) - p_hat * np.log2(p[:, i_block, 1])
                        #KL_ += (1-p_hat) * np.log2(1-p_hat) - (1-p_hat) * np.log2(1-p[:, i_block, 1])
                        #KL[i_mode, i_scan, i_block] = np.mean(KL_)
            #---------------------------------------------------------------------------
        elif TD is True : figsize = (fig_width, ((nb_fig)*fig_width)/(1.6180))
        else:             figsize=(fig_width, nb_fig*(fig_width)/1.6180)


        fig = plt.figure(figsize=figsize)#, sharex=True)
        gs = gridspec.GridSpec(nb_fig, 1, height_ratios=height_ratios, hspace=0.5)

        for x, m in enumerate(mode) :
            if N_scan>0: #show_trial is False :
                gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[x], width_ratios=[2,(1.6180)/2], wspace=0.3, hspace=0.05)
                ax1 = plt.Subplot(fig, gs1[0, 0])
                ax2 = plt.Subplot(fig, gs1[1, 0])
                ax3 = plt.Subplot(fig, gs1[:, 1])
                fig.add_subplot(ax3)

                if show_title is True : ax1.set_title('Mode %s Block %s'%(m, (block+1)), x=0.5, y=1.05, fontsize=t_label)

            else: #if show_trial is True :
                gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[x], hspace=0.05)
                ax1 = plt.Subplot(fig, gs1[0])
                ax2 = plt.Subplot(fig, gs1[1])

                if TD is True :
                    gs0 = gridspec.GridSpec(1, 1)
                    gs0.update(left=0, bottom=0.85, right=1, top=1.-0.1, hspace=0.05)
                    ax0 = plt.Subplot(fig, gs0[0]) #plt.subplot(gs1[0])
                    fig.add_subplot(ax0)

                    for card in ['bottom', 'top', 'right']: ax0.spines[card].set_visible(False)
                    ax0.spines['left'].set_bounds(0, 1)
                    ax0.set_xticks(())
                    ax0.set_xlim(-1, 200)
                    #ax0.yaxis.set_ticks_position('right')
                    ax0.set_yticks([0,1])
                    ax0.set_yticklabels(['left', 'right'], fontsize=t_label/1.8)

                    a0_pos = ax0.get_position().bounds

                    a0_pos = ax0.get_position().bounds


            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            #---------------------------------------------------------------------------
            # affiche la proba réel et les mouvements de la cible
            #---------------------------------------------------------------------------
            o = p[:, block, 0]
            p_true = p[:, block, 1]
            time = np.arange(N_trials)

            if TD is True :
                ax0.set_ylabel('TD', fontsize=t_label/1.5)
                ax0.plot(np.arange(N_trials), o, 'k.', ms=2, label='TD')
                ax0.step(range(N_trials), o, lw=1, alpha=alpha[0][1], c='k')
                ax0.fill_between(range(N_trials), np.zeros_like(o), o, lw=0, alpha=alpha[0][0], facecolor=color[0][0], step='pre')
                ax0.fill_between(range(N_trials), np.ones_like(o), o, lw=0, alpha=alpha[0][1], facecolor=color[0][1], step='pre')

            else :
                ax1.plot(np.arange(N_trials)-.5, o, 'k.', ms=4, label='TD')
                ax1.step(range(N_trials), o, lw=lw, alpha=.15, c='k')
                ax1.fill_between(range(N_trials), np.zeros_like(o), o, lw=0, alpha=alpha[0][1], facecolor=color[0][0], step='pre')

            ax1.step(range(N_trials), p_true, lw=lw, alpha=1, c=color[1][0], label=r'$P_{true}$')
            ax1.fill_between(range(N_trials), np.zeros_like(p_true), p_true, lw=0, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
            ax1.fill_between(range(N_trials), np.ones_like(p_true), p_true, lw=0, alpha=alpha[1][1], facecolor=color[1][1], step='pre')

            ax2.plot(np.arange(N_trials), np.ones_like(o)*40, c=color[1][0], alpha=0.9, lw=lw)

            #---------------------------------------------------------------------------
            # P_HAT
            #---------------------------------------------------------------------------
            if pause is not None :
                liste = [0,50,100,150,200]
                name=True
                for a in range(len(liste)-1) :
                    if not mode_compare is None:
                        ax1, ax2 = plot_result_bcp(ax1, ax2, mode_compare, p[liste[a]:liste[a+1], block, 0], np.arange(liste[a], liste[a+1]),
                                              c=c_compare, label=label_comp_bcp, name=name, show_run_lengths=False)
                    ax1, ax2 = plot_result_bcp(ax1, ax2, m, p[liste[a]:liste[a+1], block, 0], np.arange(liste[a], liste[a+1]), c=c_mode, label=label_bcp, name=name)

                    name=False
                for a in [ax1, ax2]:
                    a.bar(50, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.5, linewidth=0)
                    a.bar(100, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.5, linewidth=0)
                    a.bar(150, 140 + 2*(.05*140), bottom=-.05*140, color='k', width=.5, linewidth=0)

            else :
                if not mode_compare is None:
                    ax1, ax2 = plot_result_bcp(ax1, ax2, mode_compare, o, range(N_trials), c=c_compare, label=label_comp_bcp, show_run_lengths=False)
                ax1, ax2 = plot_result_bcp(ax1, ax2, m, o, range(N_trials), c=c_mode, label=label_bcp)


            if leg_up is True :
                if TD is True : ax1.legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 1.25, 1, 0.), loc=3, ncol=3, mode="expand", borderaxespad=0.)
                else :          ax1.legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 2.1, 1, 0.), loc=3, ncol=3, mode="expand", borderaxespad=0.)
            else :              ax1.legend(loc=(0.15, 0.55), ncol=2)#'best')
            # ax2.legend('best')
            #---------------------------------------------------------------------------
            # affiche SCORE
            #---------------------------------------------------------------------------
            if N_scan>0: #show_trial is False :
                ax3.plot(hs, np.mean(score[x, ...], axis=1), c='r', label=m)
                ax3.fill_between(hs,np.std(score[x, ...], axis=1)+np.nanmean(score[x, ...], axis=1), -np.std(score[x, ...], axis=1)+np.nanmean(score[x, ...], axis=1),
                                    lw=.5, alpha=.2, facecolor='r', step='mid')

                #ax3.vlines(h, ymin=np.nanmin(np.nanmean(score, axis=(0))), ymax=np.nanmax(np.nanmean(score, axis=(0))), lw=2, label='true')
                ax3.vlines(h, ymin=np.nanmin(score), ymax=np.nanmax(score), lw=2, label='true')
                ax3.set_xscale("log")

                ax3.set_xlabel('Hazard rate', fontsize=t_label/1.2)
                ax3.set_ylabel('Mean log-likelihood (bits)', fontsize=t_label/1.2)
                ax3.legend(frameon=False, loc="lower left")

            #---------------------------------------------------------------------------
            # cosmétique
            #---------------------------------------------------------------------------
            for a, size in zip([ax1, ax2], [1, 80]) :
                a.axis('tight')
                a.tick_params(labelsize=t_label/1.8, bottom=True, left=True)

                a.set_xlim(-1, N_trials)
                a.set_ylim(-.05*size, size + (.05*size))
                a.set_yticks(np.arange(0, size + (.05*size), size/2))



            ax1.set_ylabel('Probability', fontsize=t_label/1.5)
            ax1.set_xticks([])

            ax2.set_ylabel('Run-length', fontsize=t_label/1.5) #belief on r=p(r)
            ax2.set_xlabel('Trial #', fontsize=t_label/1.5);
            ax2.set_xticks([0, 50, 100, 150, 200])

            if m == 'expectation' : title = 'expectation $\sum_{r=0}^\infty r \cdot p(r) \cdot \hat{p}(r) $'
            elif m == 'max' : title = '$\hat{p} ( \mathrm{ArgMax}_r (p(r)) )$'
            elif m == 'mean' : title = 'mean equation'
            elif m == 'fixed' : title = 'fixed equation'
            elif m == 'leaky' : title = 'leaky equation'
            elif m == 'hindsight' : title = 'hindsight equation'
            if show_trial is True:
                ax2.bar(trial, 140 + (.05*140)+.05*140, bottom=-.05*140, color='firebrick', width=.5, linewidth=0, alpha=1)
                if show_title is True : ax1.set_title('Bayesian change point : %s'%title, x=0.5, y=1.05, fontsize=t_titre)

            #---------------------------------------------------------------------------

        #------------------------------------------------
        # Belief on r for trial view_essai
        #------------------------------------------------
        if show_trial is True :
            ax = plt.Subplot(fig, gs[-1])
            fig.add_subplot(ax)

            p_bar, r_bar, beliefs = bcp.inference(o, h=h, p0=p0, r0=r0)
            r_essai = (beliefs[:, trial])

            ax.plot(r_essai, c='k')
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')

            ax.set_xscale('log')
            ax.set_xlim(0, max_run_length)

            ax.set_xlabel('r$_{%s}$'%(trial), fontsize=t_label/1.2)
            ax.set_ylabel('p(r) at trial $%s$'%(trial), fontsize=t_label/1.5)
            if show_title is True : ax.set_title('Belief on r for trial %s'%(trial), x=0.5, y=1., fontsize=t_titre/1.5)

            ax.tick_params(labelsize=t_label/1.8, bottom=True, left=True)

        gs.tight_layout(fig)
        if TD is True :
            a1_pos = ax1.get_position().bounds
            ax0.set_position([a1_pos[0], a1_pos[1]+a1_pos[3]+0.02, a1_pos[2], a0_pos[3]])

        plt.show()



        if N_scan>0:      return fig, ax1, ax2, ax3
        elif TD is True : return fig, ax0, ax1, ax2
        else:             return fig, ax1, ax2


    def comparison(self, ax=None, proba='bcp', result='bet', mode_bcp='mean', show='kde', conditional_kde=True,
                    nb_point_kde=300j, color_kde='Greys', alpha=1, hatch=None, hatches=None, hatch_symbol = '/', levels=None,
                    t_titre=35, t_label=25, titre=None, pause=True, color_r='r', pos_r='right', line_r=True, fig=None, fig_width=15) :

        if fig is not None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width)) #/(1.6180)))

        #colors = ['black','dimgrey','grey','darkgrey','silver','rosybrown','lightcoral','indianred','firebrick','brown','darkred','red']
        nb_sujet = len(self.subjects)
        full = Analysis.Full_list(self, modes_bcp=mode_bcp, pause=pause)

        if proba == 'true' :
            ax.set_xlabel('$P_{true}$', fontsize=t_label/1)
            proba = 'proba'
        else :
            ax.set_xlabel('$\hat{P}_{%s}$'%(mode_bcp), fontsize=t_label/1)
            proba = 'p_hat_'+mode_bcp
        full_proba = full[proba]

        xmin, xmax = -0.032, 1.032

        if result=='bet' :
            res, ymin, ymax = 'results', -0.032, 1.032
            ax.set_ylabel('Probability Bet', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Probability Bet", fontsize=t_titre/1.2, x=0.5, y=1.05)


        elif result=='acceleration' :
            res, ymin, ymax = 'aa', -21.28, 21.28
            ax.set_ylabel('Acceleration of anticipation (°/s$^2$)', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Acceleration", fontsize=t_titre/1.2, x=0.5, y=1.05)

        elif result=='velocity' :
            res, ymin, ymax = 'va', -10.64, 10.64
            ax.set_ylabel('Velocity of anticipation (°/s)', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Velocity", fontsize=t_titre/1.2, x=0.5, y=1.05)

        full_result = full[res]

        if show=='scatter' :
            '''for x, color in enumerate(colors[:nb_sujet]):
                s = self.BET[x]['observer']
                ax.scatter(full_proba[full.sujet==s], full_result[full.sujet==s], c=color, alpha=0.5, linewidths=0)'''
            alpha=0.2
            for x in range(nb_sujet):
                s = self.subjects[x]
                ax.scatter(full_proba[full.sujet==s], full_result[full.sujet==s], c=[(0+(1/nb_sujet)*x, 0, 0, alpha)], linewidths=0)

        # masque les essais qui où full_result = NAN
        proba = np.ma.masked_array(full_proba.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()
        data = np.ma.masked_array(full_result.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()

        if show=='kde':
            from scipy import stats

            values = np.vstack([proba, data])
            kernel = stats.gaussian_kde(values)
            xx, yy = np.mgrid[xmin:xmax:nb_point_kde, ymin:ymax:nb_point_kde]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = np.reshape(kernel(positions).T, xx.shape)

            if conditional_kde is True :
                #print(f.sum(axis=0))
                f = f / f.sum(axis=1)[:, np.newaxis]

                #fmean = []
                #for x in range(len(f)):
                #    fmean.append([])
                #    for y in range(len(f[x])):
                #        fmean[x].append(f[x][y]/np.sum(f[x]))

            if levels is None : levels = 7
            if type(levels)==int : level=levels ;  nb_level=levels+2
            else :
                nb_level = len(levels)
                level = [(round(float(x[:-1])*np.max(f)/100, 4) if x[-1]=='%'
                         else round(float(x)*np.max(f), 4)) if type(x)==str
                         else x for x in levels]


            if hatch is True :

                from matplotlib import rcParams
                from matplotlib.colors import to_rgba
                rcParams['hatch.linewidth'] = 1.5
                rcParams['hatch.color'] = to_rgba(color_kde, alpha=alpha)

                if hatches is None : hatches = [None]*(nb_level-4) + [hatch_symbol, hatch_symbol*3, hatch_symbol*100]

                A = ax.contourf(xx, yy, f, levels=level, hatches=hatches, colors='none')
                ax.contour(xx, yy, f, levels=level, colors=color_kde, alpha=alpha)

                artists, l = A.legend_elements()

                if type(level) == list :
                    #l = ['%s (%s) < kde $\\leq$ %s (%s)'%(levels[x], level[x], levels[x+1], level[x+1])
                    #     if type(levels[x])==str else '%s < kde $\\leq$ %s'%(level[x], level[x+1])
                    #     for x in range(len(level)-1)]
                    l = ['%s < kde $\\leq$ %s'%(levels[x], levels[x+1])
                         if type(levels[x])==str else '%s < kde $\\leq$ %s'%(level[x], level[x+1])
                         for x in range(len(level)-1)]

                else :
                    l = ['%s < kde $\\leq$ %s'%(l[x][1:].split(' <')[0], l[x][:-1].split('\\leq ')[1])
                         for x in range(len(l))]

                #hist, x_edges, y_edges = np.histogram2d(proba, data ,bins=20)

                legend = ax.legend(artists, l, loc='%s'%pos_r, fontsize=t_label/1.8, frameon=True, framealpha=0.5,
                                   title='$\hat{P}_{%s}$, MI = %0.3f'%(mode_bcp, mutual_information(proba, data)),
                                   title_fontsize=t_label/1.8)

                ax.add_artist(legend)
                ax.set_xlabel('$\hat{P}$', fontsize=t_label/1)


            else :
                ax.contourf(xx, yy, f, cmap=color_kde, levels=level, alpha=alpha)

        if hatch is not True :
            ax = regress(ax, proba, data, ymin, ymax, t_label, color=color_r, pos=pos_r, line=line_r)

        if titre is not None : ax.set_title(titre, fontsize=t_titre/1.2, x=0.5, y=1.05)
        ax.axis([xmin, xmax, ymin, ymax])

        ax.tick_params(labelsize=t_label/1.8, bottom=True, left=True)
        #------------------------------------------------

        if fig is not None: return fig, ax
        else : return ax

    def plot_results(self, mode_bcp='mean', show='scatter', conditional_kde=True, tau=40., sujet=[6], fig_width=15, t_titre=35, t_label=25, plot='Full', pause=True,
                      color = [['k', 'k'], ['r', 'r'], ['k','w']], alpha = [[.35,.15],[.35,.15],[1,0]], color_bcp='darkred') :

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import bayesianchangepoint as bcp
        from scipy import stats

        nb_sujet = len(self.subjects)
        #full = Analysis.Full_list(self, modes_bcp=mode_bcp, pause=True) #(self.BET, self.EYEMVT, P_HAT=True)

        if plot == 'Full' :
            a = len(sujet)
            b = len(sujet)+1
            print('sujet+scatterKDE')
            fig, axs = plt.subplots(len(sujet)+2, 1, figsize=(fig_width, fig_width/(1.6180)))
            fig.subplots_adjust(left = 0, bottom = -1/2+(((len(sujet))*2/3)-0.16), right = 1, top =len(sujet))

            gs1 = gridspec.GridSpec(len(sujet), 1)
            gs1.update(left=0, bottom=(len(sujet))*2/3, right=1, top=len(sujet), hspace=0.05)
            for s in range(len(sujet)) : axs[s] = plt.subplot(gs1[s])

            gs2 = gridspec.GridSpec(1, 2)
            gs2.update(left=0, bottom=-1/2+(((len(sujet))*2/3)-0.16), right=1, top=((len(sujet))*2/3)-0.16, wspace=0.2)
            axs[len(sujet)] = plt.subplot(gs2[0])
            axs[len(sujet)+1] = plt.subplot(gs2[1])

        elif plot == 'sujet' :
            print('sujet')
            fig, axs = plt.subplots(len(sujet), 1, figsize=(fig_width, fig_width/(1.6180)))
            # fig.subplots_adjust(left = 0, bottom = (len(sujet))*2/3, right = 1, top =len(sujet))
            # plt.subplots_adjust(hspace=0.05)

        elif plot == 'scatterKDE' :
            a, b = 0, 1
            print('scatterKDE')
            fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_width/2.))
            # fig.subplots_adjust(left = 0, bottom = 1/2, right = 1, top =1)
            # fig.subplots_adjust(wspace=0.2)
            # plt.suptitle('Results bayesian change point %s'%(mode_bcp), fontsize=t_titre, x=0.5, y=1.3)

        if plot in ['Full', 'sujet'] :
            # axs[0].set_title('Results bayesian change point %s'%(mode_bcp), fontsize=t_titre, x=0.5, y=1.3)
            for s in range(len(sujet)) : axs[s] = results_sujet(self, axs[s], sujet, s, mode_bcp, tau, t_label, pause, color, alpha, color_bcp)

        if plot in ['Full', 'scatterKDE'] :

            #------------------------------------------------
            # SCATTER KDE Plot
            #------------------------------------------------
            if show=='kde' : color_r='r'
            else : color_r='k'

            opt = {'show':show, 'mode_bcp':mode_bcp, 'conditional_kde':conditional_kde, 't_titre':t_titre, 't_label':t_label,
                    'pause':pause, 'color_r':color_r}
            axs[a] = Analysis.comparison(self, ax=axs[a], result='bet', **opt)
            axs[b] = Analysis.comparison(self, ax=axs[b], result='velocity', **opt)

            #------------------------------------------------

        for i_layer in range(len(axs)) :
            axs[i_layer].tick_params(labelsize=t_label/1.8, bottom=True, left=True)
            if not plot in ['scatterKDE'] :
                if i_layer < len(sujet)-1 :
                    axs[i_layer].set_xticks([])
                elif i_layer == len(sujet)-1 :
                    axs[i_layer].set_xlabel('Trials', fontsize=t_label)
                    axs[i_layer].set_xticks([49, 99, 149])
                    axs[i_layer].set_xticklabels([50, 100, 150], ha='left',fontsize=t_label/1.8)

        if not plot in ['scatterKDE'] :
            axs[0].legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 1.05, 1, 0.), loc=4, ncol=4,
                           mode="expand", borderaxespad=0.)

        #------------------------------------------------

        return fig, axs





    def plot_experiment(self, sujet=[0], mode_bcp=None, tau=40, p=None, num_block=None,
                        fig=None, axs=None, fig_width=15, titre=None, t_titre=35, t_label=25,
                        color=[['k', 'k'], ['r', 'r'], ['k','w']], color_bet='r', color_va='k',
                        color_bcp=['darkgreen'], name_bcp=['$P_{BBCP}$'],
                        alpha = [[.35,.15],[.35,.15],[1,0]], lw = 1.3, pause=50, ec = 0.2):

        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy import stats
        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']
        h = 1./tau
        if mode_bcp=='leaky' : fixed_window_size=tau
        else :                 fixed_window_size = 40


        p = self.param_exp['p']
        ec = 0.1

        def plot_result_bcp(ax1, mode, observation, time, n_trial, name_bcp, color, i_block, ec, name=True) :

            from scipy.stats import beta
            p_bar, r_bar, beliefs = bcp.inference(observation, h=h, p0=p0, r0=r0)
            p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=mode, fixed_window_size=fixed_window_size, p0=p0)
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)

            for i_trial in range(n_trial):
                p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
            ax1.plot(time, i_block+p_hat+ec*i_block, c=color, lw=lw, alpha=.9, label=name_bcp if name is True else '')
            ax1.plot(time, i_block+p_sup+ec*i_block, c=color, lw=lw, alpha=.9, ls='--')#, label='CI' if name is True else '')
            ax1.plot(time, i_block+p_low+ec*i_block, c=color, lw=lw, alpha=.9, ls='--')
            ax1.fill_between(time, i_block+p_sup+ec*i_block, i_block+p_low+ec*i_block, lw=.5, alpha=.11, facecolor=color)

            return ax1


        if fig is None:
            nb_ax=len(sujet)

            import matplotlib.gridspec as gridspec
            #------------------------------------------------
            fig, axs = plt.subplots(3+1, 1, figsize=(fig_width, ((3+0.5)*fig_width/3)/(1.6180)))

            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(left=0+0.072, bottom=0.84, right=1-0.04, top=1.-0.11, hspace=0.05)
            axs[0] = plt.subplot(gs1[0])

            axs[0].plot(np.arange(1, N_trials), p[1:, num_block[0], 0], 'k.', ms=4)
            for card in ['bottom', 'top', 'right']: axs[0].spines[card].set_visible(False)
            axs[0].spines['left'].set_bounds(0, 1)

            gs2 = gridspec.GridSpec(3, 1)
            gs2.update(left=0+0.072, bottom=0+0.1, right=1-0.04, top=0.85-0.03, hspace=0.05)
            for a in range(3): axs[a+1] = plt.subplot(gs2[a])

        for i_layer in range(len(axs)):
            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            for num_pause in range(1,4) : axs[i_layer].bar(num_pause*pause-1, len(num_block)+ec*len(num_block), bottom=-ec/2, color='k', width=.5, linewidth=0)

            #------------------------------------------------
            # cosmétique
            #------------------------------------------------
            axs[i_layer].set_ylim(-(ec/2), len(num_block) +ec*len(num_block)-(ec/2))
            y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
            axs[i_layer].set_yticks(y_ticks[:len(num_block)*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('right')

            axs[i_layer].set_xlim(-1, N_trials)
            if i_layer==(len(axs)-1) :
                axs[i_layer].set_xticks([0, 50, 100, 150, 200])
                axs[i_layer].set_xticklabels([0, 50, 100, 150, 200], ha='left', fontsize=t_label/1.8)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])
        axs[0].yaxis.set_ticks_position('left')
        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
        axs[0].set_yticks(y_ticks[:len(num_block)*2])
        axs[0].set_yticklabels(['left','right']*len(num_block),fontsize=t_label/1.8)
        ###################################################################################################################################

        td_label = 'TD'

        for i_block, block in enumerate(num_block):

            #print(num_block, i_block, block, i_block+ec*i_block)
            axs[0].step(range(N_trials), p[:, block, 0]+i_block+ec*i_block, lw=1, c=color[0][0], alpha=alpha[0][0], where='mid')
            axs[0].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 0])+ec*i_block,
                                      i_block+p[:, block, 0]+ec*i_block,
                                      lw=.5, alpha=alpha[0][0], facecolor=color[0][0], step='mid')
            axs[0].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 0])+ec*i_block,
                                      i_block+p[:, block, 0]+ec*i_block,
                                      lw=.5, alpha=alpha[0][1], facecolor=color[0][1], step='mid')
            axs[0].set_ylabel(td_label, fontsize=t_label/1.2)

            for a in range(1, 4) :
                axs[a].step(range(N_trials), p[:, block, 1]+i_block+ec*i_block, lw=lw, c=color[1][0], alpha=1, label='$P_{true}$')
                axs[a].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                          lw=.5, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
                axs[a].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                          lw=.5, alpha=alpha[1][1], facecolor=color[1][1], step='pre')

                axs[a].plot(range(N_trials), 0.5*np.ones(N_trials)+i_block+ec*i_block, lw=1.5, c='k', alpha=0.5)
                #axs[a].text(-0.055, 0.5, 'Subject %s'%(s), fontsize=t_label/1.2, rotation=90, transform=axs[a].transAxes, ha='right', va='center')
        #-------------------------------------------------------------------------------------------------------------

        p = self.BET[self.subjects[0]]['p']
        mini = 5 #8
        ec1 = ec*mini*2
        p0, r0 =  0.5, 1.0
        ax2 = axs[2].twinx()

        for i_block, block in enumerate(num_block):

            resusu = np.zeros((len(self.subjects), N_trials))
            v_antiti = np.zeros((len(self.subjects), N_trials))

            for x, y in enumerate(self.subjects) :
                resusu[x] = (self.BET[y]['results'][:, block]+1)/2
                a_anti, start_anti, latency = self.EYEMVT[y]['a_anti'][block], self.EYEMVT[y]['start_anti'][block], self.EYEMVT[y]['latency'][block]
                v_antiti[x] = (np.array(a_anti)*((np.array(latency)-np.array(start_anti))/1000))

            results = np.median(resusu, axis=0)
            va = np.median(v_antiti, axis=0)

            results_low, results_sup = np.percentile(resusu, [25, 75], axis=0)
            va_low, va_sup = np.percentile(v_antiti, [25, 75], axis=0)

            #------------------------------------------------------------
            # BCP
            #------------------------------------------------------------

            for b, mode in enumerate(mode_bcp):
                liste = [0,50,100,150,200]
                for pause in range(len(liste)-1) :
                    if pause==0 : name=True
                    else :        name=False
                    n_trial = liste[pause+1]-liste[pause]
                    axs[1] = plot_result_bcp(axs[1], mode, p[liste[pause]:liste[pause+1], block, 0],
                                             np.arange(liste[pause], liste[pause+1]), n_trial, name_bcp[b], color_bcp[b], i_block, ec, name=name)

            liste = [0,50,100,150,200]
            for pause in range(len(liste)-1) :

                #------------------------------------------------------------
                # eyeMvt
                #------------------------------------------------------------
                va__ = 2*(mini*i_block)+va[liste[pause]:liste[pause+1]]+ec1*i_block
                va_sup__ = 2*(mini*i_block)+va_sup[liste[pause]:liste[pause+1]]+ec1*i_block
                va_low__ = 2*(mini*i_block)+va_low[liste[pause]:liste[pause+1]]+ec1*i_block
                titi = np.arange(liste[pause], liste[pause+1])

                #axs[2].step(range(1), -1000, color=color_va, lw=lw, alpha=1, label='Eye movement'  if i_block==0 else '')
                ax2.plot(titi, va__, color=color_va, lw=lw, alpha=1)#, label='Eye movement' if i_block==0 else '')

                ax2.fill_between(titi, va_sup__, va_low__, lw=.5, alpha=.3, facecolor=color_va)#, step='pre')
                ax2.plot(titi, va_sup__, c=color_va, lw=lw, alpha=.9, ls='--')
                ax2.plot(titi, va_low__, c=color_va, lw=lw, alpha=.9, ls='--')

                #------------------------------------------------------------
                # bet
                #------------------------------------------------------------
                results__ = i_block+results[liste[pause]:liste[pause+1]]+ec*i_block
                results_sup__ = i_block+results_sup[liste[pause]:liste[pause+1]]+ec*i_block
                results_low__ = i_block+results_low[liste[pause]:liste[pause+1]]+ec*i_block

                axs[3].plot(titi, results__, color=color_bet, lw=lw, alpha=1)#, label='Individual guess'  if i_block==0 else '')

                axs[3].fill_between(titi, results_sup__, results_low__, lw=.5, alpha=.3, facecolor=color_bet)#, step='pre')
                axs[3].plot(titi, results_sup__, c=color_bet, lw=lw, alpha=.9, ls='--')
                axs[3].plot(titi, results_low__, c=color_bet, lw=lw, alpha=.9, ls='--')


            for c, suj in zip(['b', 'g'], sujet) :

                results = (self.BET[self.subjects[suj]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
                a_anti, start_anti, latency = self.EYEMVT[self.subjects[suj]]['a_anti'], self.EYEMVT[self.subjects[suj]]['start_anti'], self.EYEMVT[self.subjects[suj]]['latency']
                print('sujet', suj, '=', self.subjects[suj])
                # eyeMvt
                #------------------------------------------------------------
                #axs[3].step(range(1), -1000, color=color_va, lw=lw, alpha=1)#, label='Eye movement'  if i_block==0 else '')
                va = (np.array(a_anti[block])*((np.array(latency[block])-np.array(start_anti[block]))/1000))
                ax2.step(range(N_trials), 2*(mini*i_block)+va+ec1*i_block, color=c, lw=lw, alpha=.7)
                # bet
                #------------------------------------------------------------
                axs[3].step(range(N_trials), i_block+results[:, block]+ec*i_block, lw=lw, alpha=.7, color=c)#, label='Individual guess'  if i_block==0 else '')


        axs[1].set_ylabel('Probability', fontsize=t_label/1.5)
        axs[1].yaxis.set_ticks_position('left')
        y_ticks=[0, 0.5, 1, 1+ec, 1.5+ec, 2+ec, 2+ec*2, 2.5+ec*2, 3+ec*2]
        axs[1].set_yticks(y_ticks[:len(num_block)*3])
        axs[1].set_yticklabels(['0', '0.5', '1']*len(num_block),fontsize=t_label/1.8)
        axs[1].yaxis.set_label_coords(-0.03, 0.5)


        #----------------------------------------------------------------
        axs[2].set_yticks([])

        ax2.set_ylim(-mini-(ec1/2), len(num_block)*mini + ec1*len(num_block)-(ec1/2))
        y_ticks=[-mini, 0, mini,
                 mini+ec1, 2*mini+ec1, 3*mini+ec1,
                 3*mini+2*ec1, 4*mini+2*ec1, 5*mini+2*ec1]

        ax2.set_yticks(y_ticks[:len(num_block)*3])
        ax2.set_yticklabels(['-%s'%mini, '0', '%s'%mini]*len(num_block),fontsize=t_label/1.8)
        ax2.yaxis.set_label_coords(-0.063, 0.5) #(-0.053, 0.5)
        #ax2.yaxis.set_tick_params(direction='out', colors=color_va)
        ax2.yaxis.set_ticks_position('left')
        ax2.set_ylabel('Anticipatory\neye velocity (°/s)', fontsize=t_label/1.5)#, color=color_va) # 'Velocity of eye °/s'

        #----------------------------------------------------------------
        axs[3].yaxis.set_ticks_position('left')
        y_ticks=[0, 0.5, 1, 1+ec, 1.5+ec, 2+ec, 2+ec*2, 2.5+ec*2, 3+ec*2]
        axs[3].set_yticks(y_ticks[:len(num_block)*3])
        axs[3].set_yticklabels(['0', '0.5', '1']*len(num_block),fontsize=t_label/1.8)

        axs[3].set_ylabel('Bet score', fontsize=t_label/1.5)#, color=color_bet)
        #axs[3].tick_params('y', colors=color_bet)
        axs[3].yaxis.set_label_coords(-0.03, 0.5)


        #------------------------------------------------
        #y_t = 1.25
        #axs[0].set_title(titre, fontsize=t_titre, x=0.5, y=y_t)

        ncol_leg = 4
        axs[1].legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 1.33, 1, 0.), loc=3, ncol=ncol_leg, mode="expand", borderaxespad=0.)

        axs[-1].set_xlabel('Trial #', fontsize=t_label)
        try: fig.tight_layout()
        except: print('tight_layout failed :-(')
        plt.subplots_adjust(hspace=0.05)
        #------------------------------------------------

        return fig, axs

    def comparison_line(self, ax=None, result='bet', mode_bcp=['real', 'leaky', 'mean'], bins=[0,0.2,0.4,0.6,0.8,1],
                        alpha=1, t_titre=35, t_label=25, titre=None, pause=True, color_r=['b', 'tab:orange', 'g'],
                        fig=None, fig_width=15, offset=0.01, lw_r=2) :

        if fig is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width)) #/(1.6180)))

        nb_sujet = len(self.subjects)

        modes_bcp, name_mode = [],[]
        for mode in mode_bcp :
            if mode == 'real' :
                name_mode.append('$P_{true}$')
            else :
                modes_bcp.append(mode)
                if mode== 'mean' : name_mode.append('$P_{BBCP}$')
                else : name_mode.append('$P_{%s}$'%(mode))
        full = Analysis.Full_list(self, modes_bcp=modes_bcp, pause=pause)

        xmin, xmax = -0.032, 1.032

        if result=='bet' :
            res, ymin, ymax = 'results', -0.032, 1.032
            ax.set_ylabel('Bet score', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Probability Bet", fontsize=t_titre/1.2, x=0.5, y=1.05)

        elif result=='acceleration' :
            res, ymin, ymax = 'aa', -21.28, 21.28
            ax.set_ylabel('Acceleration of anticipation (°/s$^2$)', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Acceleration", fontsize=t_titre/1.2, x=0.5, y=1.05)

        elif result=='velocity' :
            res, ymin, ymax = 'va', -5.32, 5.32 # -10.64, 10.64 # #
            #ax.set_ylabel('Velocity of anticipation (°/s)', fontsize=t_label/1.2)
            ax.set_ylabel('Anticipatory eye velocity (°/s)', fontsize=t_label/1.2) # 'Velocity of eye (°/s)'
            if titre is None : ax.set_title("Velocity", fontsize=t_titre/1.2, x=0.5, y=1.05)

        full_result = full[res]

        a1 = fig.add_axes([0, 1, 0.25, 0.25])
        a2 = fig.add_axes([1, 0, 0.25, 0.25])
        a1.set_title("$r^{2}$", fontsize=t_label/1.8)
        a2.set_title("MI", fontsize=t_label/1.8)
        a1.set_yticks([0, 0.5, 1.])
        a1.set_ylim(0, 1.5)
        a1.spines['left'].set_bounds(0, 1)

        a2.set_yticks([0., 1, 2])
        a2.set_ylim(0, 3)
        a2.spines['left'].set_bounds(0, 2)

        for a in [a1, a2] :
            a.set_xticks([])
            #a.set_xticklabels([]) #a.set_xticklabels(name_mode)
            a.tick_params(labelsize=t_label/2.7)
            for card in ['top', 'right']: a.spines[card].set_visible(False)
            a.set_xlim(-0.6, 2.5)
            a.spines['bottom'].set_bounds(-0.5, 2.5)

        x_bin = np.linspace(-offset, offset, len(mode_bcp))

        R_i, MI_i = [], []
        for i, mode in enumerate(mode_bcp) :
            if mode == 'real' : proba = 'proba'
            else :              proba = 'p_hat_'+mode
            full_proba = full[proba]

            proba = np.ma.masked_array(full_proba.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()
            data = np.ma.masked_array(full_result.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()


            meme, bibi = [], []
            for b in range(len(bins)-1) :
                d = [data[x] for x in range(len(proba)) if proba[x]>=bins[b] and proba[x]<bins[b+1]]
                yerr_l, yerr_s = np.percentile(d, [25, 75])
                #ax.errorbar((bins[b+1]+bins[b])/2+x_bin[i], np.mean(d), yerr=np.std(d), color=color_r[i], capsize=10, markersize=10, marker='o')
                ax.errorbar((bins[b+1]+bins[b])/2+x_bin[i], yerr_l, yerr=[[0], [yerr_s-yerr_l]], color=color_r[i], capsize=(1000*offset)/3)
                ax.scatter((bins[b+1]+bins[b])/2+x_bin[i], np.median(d), color=color_r[i], s=100, marker='o')
                ax.bar((bins[b+1]+bins[b])/2+x_bin[i], yerr_s-yerr_l, offset, yerr_l, color=color_r[i], alpha=.4)

                bibi.append((bins[b+1]+bins[b])/2+x_bin[i])
                meme.append(np.median(d))

            ax.plot(bibi, meme, lw=6, alpha=.4, color=color_r[i])
            #for t in [0.01,0.2,0.4,0.6,0.7,0.8,1] : ax.vlines(t, 0,1, alpha=.2) ; ax.hlines(t, 0,1, alpha=.2)

            R_s, MI_s = [], []
            for j, s in enumerate(self.subjects):
                proba_s = np.ma.masked_array(full_proba[full.sujet==s].values.tolist(), mask=np.isnan(full_result[full.sujet==s].values.tolist())).compressed()
                data_s = np.ma.masked_array(full_result[full.sujet==s].values.tolist(), mask=np.isnan(full_result[full.sujet==s].values.tolist())).compressed()
                ax, r_s, mi_s = regress(ax, proba_s, data_s, line=False, text=False, return_r_mi=True)
                r_s = r_s**2
                R_s.append(r_s)
                MI_s.append(mi_s)
                #a1.scatter(i, r_s, c=color_r[i], linewidths=0, marker='_', s=5500)
                #a2.scatter(i, mi_s, c=color_r[i], linewidths=0, marker='_', s=5500)
                a1.scatter(i-0.4+(0.8*(j/(len(self.subjects)-1))), r_s, c=color_r[i], linewidths=0, marker='o', s=50)
                a2.scatter(i-0.4+(0.8*(j/(len(self.subjects)-1))), mi_s, c=color_r[i], linewidths=0, marker='o', s=50)

            #ax, r_, mi = regress(ax, proba, data, ymin, ymax, t_label, color=color_r[i], text=False, return_r_mi=True,lw=lw_r)
            #r_ = r_**2
            mi = np.median(MI_s)
            r_ = np.median(R_s)

            a1.bar(i, r_, color=color_r[i], width=0.9, alpha=0.4)
            a2.bar(i, mi, color=color_r[i], width=0.9, alpha=0.4)
            a1.text(i, 0+0.05, '%.3f'%r_, color=color_r[i], alpha=1, ha="center", va="bottom", fontsize=t_label/2.7, weight='bold')
            a2.text(i, 0+0.05, '%.3f'%mi, color=color_r[i], alpha=1, ha="center", va="bottom", fontsize=t_label/2.7, weight='bold')

            a1.text(i, 0-0.05, name_mode[i], color=color_r[i], alpha=1, ha="center", va="top", fontsize=t_label/2, weight='bold')
            a2.text(i, 0-0.05, name_mode[i], color=color_r[i], alpha=1, ha="center", va="top", fontsize=t_label/2, weight='bold')


            print(mode, ':')
            print('mean r^2 :', np.mean(R_s), '+/-', np.std(R_s))
            print('mean MI :', np.mean(MI_s), '+/-', np.std(MI_s), end='\n\n')
            R_i.append(R_s)
            MI_i.append(MI_s)

        from scipy.stats import wilcoxon

        for i  in range(len(mode_bcp)) :
            for j  in range(i+1, len(mode_bcp)) :

                print(mode_bcp[i], mode_bcp[j])
                a = (j-i)*0.1
                x_1, x_2 = i-a, j+a
                if i==1 : x_1 = i+a
                if j==1 : x_2 = j-a

                w_r = wilcoxon(R_i[i], R_i[j]) ; print('r^2 =', w_r)
                if w_r.pvalue < 0.05 :
                    a1.hlines(0.9+((j-i)*0.2), x_1, x_2)
                    #a1.vlines(x_1, np.max(R_i[i])+0.25, np.max(R_i)+((j-i)*0.25))
                    #a1.vlines(x_2, np.max(R_i[j])+0.25, np.max(R_i)+((j-i)*0.25))
                    a1.text((x_1+x_2)/2, 0.9+((j-i)*0.2), '**' if w_r.pvalue<0.01 else '*', fontsize=t_label/2.1, ha='center')

                w_mi = wilcoxon(MI_i[i], MI_i[j]) ; print('mi =', w_mi, '\n')
                if w_mi.pvalue < 0.05 :
                    a2.hlines(1.8+((j-i)*0.4), x_1, x_2)
                    #a2.vlines(x_1, np.max(MI_i[i])+0.25, np.max(MI_i)+((j-i)*0.3))
                    #a2.vlines(x_2, np.max(MI_i[j])+0.25, np.max(MI_i)+((j-i)*0.3))
                    a2.text((x_1+x_2)/2, 1.8+((j-i)*0.4), '**'  if w_mi.pvalue<0.01 else '*', fontsize=t_label/1.8, ha='center')

        if titre is not None : ax.set_title(titre, fontsize=t_titre/1.2, x=0.5, y=1.05)
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_xlabel('Probability', fontsize=t_label/1.2)
        ax.tick_params(labelsize=t_label/1.8, bottom=True, left=True)

        list_spine = [0, 0.2, 0.4, 0.6, 0.8, 1]
        list_ticks, titi = [], []
        for i, spipi in enumerate(list_spine[:-1]) :
            ax_ = ax.twinx()
            ax_.set_yticks([])
            for card in ['top', 'right', 'left']: ax_.spines[card].set_visible(False)
            ax_.spines['bottom'].set_bounds(spipi+0.03, spipi+0.17)

            list_ticks.extend([spipi+0.03, spipi+0.17])
            titi.append(spipi+0.1)

        ax.set_xticks(list_ticks)
        ax.tick_params('x', direction='in', labelcolor='w')# colors='w', labelcolor='k')#, direction='in')

        tlabel = ['0', '0.2', '0.4', '0.6', '0.8', '1']
        for i, t in enumerate(titi) :
            ax.text(t, ymin-((ymax-ymin)*0.005), '%s - %s'%(tlabel[i], tlabel[i+1]), ha="center", va="top", fontsize=t_label/1.8)#, weight='bold')


        for card in ['top', 'right','bottom']: ax.spines[card].set_visible(False)
        if result=='bet' : ax.spines['left'].set_bounds(0, 1)
        else : ax.spines['left'].set_bounds(-5, 5)


        #------------------------------------------------
        fig.tight_layout()

        ax_pos = ax.get_position().bounds
        a1.set_position([ax_pos[0]+0.055, ax_pos[1]+ax_pos[3]-0.3, 0.25, 0.25])
        a2.set_position([ax_pos[0]+ax_pos[2]-0.275, ax_pos[1]+0.055, 0.25, 0.25])



        if fig is None: return fig, ax
        else : return ax

    def comparison_h_BBCP(self, mode_bcp='mean', list_h=[1/40], p=None, num_block=None,
                   fig=None, axs=None, fig_width=15, titre=None, t_titre=35, t_label=25,
                   color=[['k', 'k'], ['r', 'r'], ['k','w']],
                   color_bcp=['darkgreen'], name_bcp=['$P_{BBCP}$'],
                   alpha = [[.35,.15],[.35,.15],[1,0]], lw = 1.3, pause=50, ec = 0.2):

        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy import stats
        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']

        p = self.param_exp['p']
        ec = 0.1

        def plot_result_bcp(ax1, h, observation, time, n_trial, name_bcp, color, i_block, ec, name=True) :

            fixed_window_size = 1/h

            from scipy.stats import beta
            p_bar, r_bar, beliefs = bcp.inference(observation, h=h, p0=p0, r0=r0)
            p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=mode_bcp, fixed_window_size=fixed_window_size, p0=p0)
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)

            for i_trial in range(n_trial):
                p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
            ax1.plot(time, i_block+p_hat+ec*i_block, c=color, lw=lw, alpha=.9)
            ax1.plot(time, i_block+p_sup+ec*i_block, c=color, lw=lw, alpha=.9, ls='--')
            ax1.plot(time, i_block+p_low+ec*i_block, c=color, lw=lw, alpha=.9, ls='--')
            ax1.fill_between(time, i_block+p_sup+ec*i_block, i_block+p_low+ec*i_block, lw=.5, alpha=.11, facecolor=color)

            return ax1


        if fig is None:

            import matplotlib.gridspec as gridspec
            #------------------------------------------------
            fig, axs = plt.subplots(len(list_h)+1, 1, figsize=(fig_width, ((3+0.5)*fig_width/3)/(1.6180)))
            #gs0 = gridspec.GridSpec(2, 1)
            #gs1, gs2 = gs0[0], gs0[1]

            gs1 = gridspec.GridSpec(1, 1)
            gs1.update(left=0+0.072, bottom=0.84, right=1-0.04, top=1.-0.11, hspace=0.05)
            axs[0] = plt.subplot(gs1[0])

            axs[0].plot(np.arange(1, N_trials), p[1:, num_block[0], 0], 'k.', ms=4)
            for card in ['bottom', 'top', 'right']: axs[0].spines[card].set_visible(False)
            axs[0].spines['left'].set_bounds(0, 1)

            gs2 = gridspec.GridSpec(len(list_h), 1)
            gs2.update(left=0+0.072, bottom=0+0.1, right=1-0.04, top=0.85-0.03, hspace=0.05)
            for a in range(len(list_h)): axs[a+1] = plt.subplot(gs2[a])

        for i_layer in range(len(axs)):
            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            if pause is not None :
                for num_pause in range(1,4) :
                    axs[i_layer].bar(num_pause*pause-1, len(num_block)+ec*len(num_block), bottom=-ec/2, color='k', width=.5, linewidth=0)

            #------------------------------------------------
            # cosmétique
            #------------------------------------------------
            axs[i_layer].set_ylim(-(ec/2), len(num_block) +ec*len(num_block)-(ec/2))
            y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
            axs[i_layer].set_yticks(y_ticks[:len(num_block)*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('right')

            axs[i_layer].set_xlim(-1, N_trials)
            if i_layer==(len(axs)-1) :
                axs[i_layer].set_xticks([0, 50, 100, 150, 200])
                axs[i_layer].set_xticklabels([0, 50, 100, 150, 200], ha='left', fontsize=t_label/1.8)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])
        axs[0].yaxis.set_ticks_position('left')
        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
        axs[0].set_yticks(y_ticks[:len(num_block)*2])
        axs[0].set_yticklabels(['left','right']*len(num_block),fontsize=t_label/1.8)
        axs[0].yaxis.set_label_coords(-0.07, 0.5)
        ###################################################################################################################################

        td_label = 'TD'

        axs[0].set_ylabel(td_label, fontsize=t_label/1.5)
        for i_block, block in enumerate(num_block):

            #print(num_block, i_block, block, i_block+ec*i_block)
            axs[0].step(range(N_trials), p[:, block, 0]+i_block+ec*i_block, lw=1, c=color[0][0], alpha=alpha[0][0], where='mid')
            axs[0].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 0])+ec*i_block,
                                      i_block+p[:, block, 0]+ec*i_block,
                                      lw=.5, alpha=alpha[0][0], facecolor=color[0][0], step='mid')
            axs[0].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 0])+ec*i_block,
                                      i_block+p[:, block, 0]+ec*i_block,
                                      lw=.5, alpha=alpha[0][1], facecolor=color[0][1], step='mid')


            for a in range(1, len(list_h)+1) :
                axs[a].step(range(N_trials), p[:, block, 1]+i_block+ec*i_block, lw=lw, c=color[1][0], alpha=1, label='$P_{true}$')
                axs[a].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                          lw=.5, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
                axs[a].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                          lw=.5, alpha=alpha[1][1], facecolor=color[1][1], step='pre')

                axs[a].plot(range(N_trials), 0.5*np.ones(N_trials)+i_block+ec*i_block, lw=1.5, c='k', alpha=0.5)
                #axs[a].text(-0.055, 0.5, 'Subject %s'%(s), fontsize=t_label/1.2, rotation=90, transform=axs[a].transAxes, ha='right', va='center')
        #-------------------------------------------------------------------------------------------------------------

        p = self.BET[self.subjects[0]]['p']
        p0, r0 =  0.5, 1.0

        for i_block, block in enumerate(num_block):

            #------------------------------------------------------------
            # BCP
            #------------------------------------------------------------

            for b, h in enumerate(list_h):

                axs[1].plot(-1000000,-100000, lw=lw, c=color_bcp[b], label='%s = 1/%s'%(name_bcp[b], int(1/h)))
                #axs[1].plot(-1000000,-100000, lw=lw, c=color_bcp[b], label='h = $10^{%.1f}$'%(np.log10(h)))


                if pause is None :
                    axs[b+1] = plot_result_bcp(axs[b+1], h, p[:, block, 0], np.arange(len(p[:, block, 0])),
                                                len(p[:, block, 0]), name_bcp[b], color_bcp[b], i_block, ec, name=True)
                    #axs[b+1].set_ylabel('h=%s'%h, fontsize=t_label/1.5)
                    #axs[b+1].tick_params('y', colors=color_bcp[b])



                else :
                    liste = [0,50,100,150,200]
                    for pause in range(len(liste)-1) :
                        if pause==0 :
                            name=True
                            #axs[b+1].set_ylabel('h=%s'%h, fontsize=t_label/1.5)
                            #axs[b+1].tick_params('y', colors=color_bcp[b])
                        else :
                            name=False
                        n_trial = liste[pause+1]-liste[pause]
                        axs[b+1] = plot_result_bcp(axs[b+1], h, p[liste[pause]:liste[pause+1], block, 0], np.arange(liste[pause], liste[pause+1]),
                                                   n_trial, name_bcp[b], color_bcp[b], i_block, ec, name=name)


        for i in range(1,len(list_h)+1) :

            #axs[i].yaxis.set_label_coords(-0.03, 0.5)
            axs[i].set_yticklabels(['0', '0.5', '1']*len(num_block),fontsize=t_label/1.8)

            axs[i].yaxis.set_ticks_position('left')
            y_ticks=[0, 0.5, 1, 1+ec, 1.5+ec, 2+ec, 2+ec*2, 2.5+ec*2, 3+ec*2]
            axs[i].set_yticks(y_ticks[:len(num_block)*3])

        ncol_leg = 4
        axs[1].legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 1.33, 1, 0.), loc=3, ncol=ncol_leg, mode="expand", borderaxespad=0.)
        axs[2].set_ylabel('Probability', fontsize=t_label/1.5)
        #----------------------------------------------------------------

        axs[-1].set_xlabel('Trial #', fontsize=t_label/1.5)


        #gs0.tight_layout(fig) #; gs2.tight_layout(fig)
        plt.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        #try: fig.tight_layout()
        #except: print('tight_layout failed :-(')
        #------------------------------------------------

        #fig.tight_layout()
        return fig, axs


    #---------------------------------------------------------------------
    def comparison_line_old(self, ax=None, result='bet', mode_bcp=['true', 'leaky', 'mean'], bins=[0,0.2,0.4,0.6,0.8,1],
                        alpha=1, t_titre=35, t_label=25, titre=None, pause=True, color_r=['b', 'tab:orange', 'g'],
                        fig=None, fig_width=15, offset=0.01, lw_r=2) :

        if fig is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_width)) #/(1.6180)))

        nb_sujet = len(self.subjects)

        modes_bcp, name_mode = [],[]
        for mode in mode_bcp :
            if mode == 'true' :
                name_mode.append('$P_{true}$')
            else :
                modes_bcp.append(mode)
                if mode== 'mean' : name_mode.append('$P_{BBCP}$')
                else : name_mode.append('$P_{%s}$'%(mode))
        full = Analysis.Full_list(self, modes_bcp=modes_bcp, pause=pause)

        xmin, xmax = -0.032, 1.032

        if result=='bet' :
            res, ymin, ymax = 'results', -0.032, 1.032
            ax.set_ylabel('Bet score', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Probability Bet", fontsize=t_titre/1.2, x=0.5, y=1.05)

        elif result=='acceleration' :
            res, ymin, ymax = 'aa', -21.28, 21.28
            ax.set_ylabel('Acceleration of anticipation (°/s$^2$)', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Acceleration", fontsize=t_titre/1.2, x=0.5, y=1.05)

        elif result=='velocity' :
            res, ymin, ymax = 'va', -5.32, 5.32 # -10.64, 10.64 # #
            #ax.set_ylabel('Velocity of anticipation (°/s)', fontsize=t_label/1.2)
            ax.set_ylabel('Velocity of eye (°/s)', fontsize=t_label/1.2)
            if titre is None : ax.set_title("Velocity", fontsize=t_titre/1.2, x=0.5, y=1.05)

        full_result = full[res]

        a1 = fig.add_axes([0, 1, 0.25,0.25])
        a2 = fig.add_axes([1, 0, 0.25, 0.25])
        a1.set_title("r", fontsize=t_label/1.8)
        a2.set_title("MI", fontsize=t_label/1.8)
        a1.set_yticks([0, 0.5, 1.])
        a1.set_ylim(0, 1.5)
        a1.spines['left'].set_bounds(0, 1)

        a2.set_yticks([0., 1, 2])
        a2.set_ylim(0, 3)
        a2.spines['left'].set_bounds(0, 2)

        for a in [a1, a2] :
            a.set_xticks([])
            #a.set_xticklabels([]) #a.set_xticklabels(name_mode)
            a.tick_params(labelsize=t_label/2.7)
            for card in ['top', 'right']: a.spines[card].set_visible(False)
            a.set_xlim(-0.6, 2.5)
            a.spines['bottom'].set_bounds(-0.5, 2.5)

        x_bin = np.linspace(-offset, offset, len(mode_bcp))

        R_i, MI_i = [], []
        for i, mode in enumerate(mode_bcp) :
            if mode == 'true' : proba = 'proba'
            else :              proba = 'p_hat_'+mode
            full_proba = full[proba]

            proba = np.ma.masked_array(full_proba.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()
            data = np.ma.masked_array(full_result.values.tolist(), mask=np.isnan(full_result.values.tolist())).compressed()

            for b in range(len(bins)-1) :
                d = [data[x] for x in range(len(proba)) if proba[x]>=bins[b] and proba[x]<bins[b+1]]
                yerr_l, yerr_s = np.percentile(d, [25, 75])
                ax.errorbar((bins[b+1]+bins[b])/2+x_bin[i], yerr_l, yerr=[[0], [yerr_s-yerr_l]], color=color_r[i], capsize=10)
                ax.scatter((bins[b+1]+bins[b])/2+x_bin[i], np.median(d), color=color_r[i], s=100, marker='o')
                #ax.errorbar((bins[b+1]+bins[b])/2+x_bin[i], np.mean(d), yerr=np.std(d), color=color_r[i], capsize=10, markersize=10, marker='o')

            ax, r_, mi = regress(ax, proba, data, ymin, ymax, t_label, color=color_r[i], text=False, return_r_mi=True,lw=lw_r)
            a1.bar(i, r_, color=color_r[i], width=0.9, alpha=0.4)
            a2.bar(i, mi, color=color_r[i], width=0.9, alpha=0.4)
            a1.text(i, 0+0.05, '%.3f'%r_, color=color_r[i], alpha=1, ha="center", va="bottom", fontsize=t_label/2.7, weight='bold')
            a2.text(i, 0+0.05, '%.3f'%mi, color=color_r[i], alpha=1, ha="center", va="bottom", fontsize=t_label/2.7, weight='bold')

            a1.text(i, 0-0.05, name_mode[i], color=color_r[i], alpha=1, ha="center", va="top", fontsize=t_label/2, weight='bold')
            a2.text(i, 0-0.05, name_mode[i], color=color_r[i], alpha=1, ha="center", va="top", fontsize=t_label/2, weight='bold')

            R_s, MI_s = [], []
            for j, s in enumerate(self.subjects):
                proba_s = np.ma.masked_array(full_proba[full.sujet==s].values.tolist(), mask=np.isnan(full_result[full.sujet==s].values.tolist())).compressed()
                data_s = np.ma.masked_array(full_result[full.sujet==s].values.tolist(), mask=np.isnan(full_result[full.sujet==s].values.tolist())).compressed()
                ax, r_s, mi_s = regress(ax, proba_s, data_s, line=False, text=False, return_r_mi=True)
                R_s.append(r_s)
                MI_s.append(mi_s)
                #a1.scatter(i, r_s, c=color_r[i], linewidths=0, marker='_', s=5500)
                #a2.scatter(i, mi_s, c=color_r[i], linewidths=0, marker='_', s=5500)
                a1.scatter(i-0.4+(0.8*(j/(len(self.subjects)-1))), r_s, c=color_r[i], linewidths=0, marker='o', s=50)
                a2.scatter(i-0.4+(0.8*(j/(len(self.subjects)-1))), mi_s, c=color_r[i], linewidths=0, marker='o', s=50)

            if mode != 'mean' : print(mode, ':')
            else :              print('BBCP :')
            print('mean r :', np.mean(R_s), '+/-', np.std(R_s))
            print('mean MI :', np.mean(MI_s), '+/-', np.std(MI_s), end='\n\n')
            R_i.append(R_s)
            MI_i.append(MI_s)

        from scipy.stats import wilcoxon

        for i  in range(len(mode_bcp)) :
            if mode_bcp[i] != 'mean' : m_1 = mode_bcp[i]
            else :                     m_1 = 'BBCP'

            for j  in range(i+1, len(mode_bcp)) :
                if mode_bcp[j] != 'mean' : m_2 = mode_bcp[j]
                else :                     m_2 = 'BBCP'

                print(m_1, m_2)
                a = (j-i)*0.1
                x_1, x_2 = i-a, j+a
                if i==1 : x_1 = i+a
                if j==1 : x_2 = j-a

                w_r = wilcoxon(R_i[i], R_i[j]) ; print('r =', w_r)
                if w_r.pvalue < 0.05 :
                    a1.hlines(0.9+((j-i)*0.2), x_1, x_2)
                    #a1.vlines(x_1, np.max(R_i[i])+0.25, np.max(R_i)+((j-i)*0.25))
                    #a1.vlines(x_2, np.max(R_i[j])+0.25, np.max(R_i)+((j-i)*0.25))
                    a1.text((x_1+x_2)/2, 0.9+((j-i)*0.2), '**' if w_r.pvalue<0.01 else '*', fontsize=t_label/2.1, ha='center')

                w_mi = wilcoxon(MI_i[i], MI_i[j]) ; print('mi =', w_mi, '\n')
                if w_mi.pvalue < 0.05 :
                    a2.hlines(1.8+((j-i)*0.4), x_1, x_2)
                    #a2.vlines(x_1, np.max(MI_i[i])+0.25, np.max(MI_i)+((j-i)*0.3))
                    #a2.vlines(x_2, np.max(MI_i[j])+0.25, np.max(MI_i)+((j-i)*0.3))
                    a2.text((x_1+x_2)/2, 1.8+((j-i)*0.4), '**'  if w_mi.pvalue<0.01 else '*', fontsize=t_label/1.8, ha='center')

        if titre is not None : ax.set_title(titre, fontsize=t_titre/1.2, x=0.5, y=1.05)
        ax.axis([xmin, xmax, ymin, ymax])
        ax.set_xlabel('Probability', fontsize=t_label/1.2)
        ax.tick_params(labelsize=t_label/1.8, bottom=True, left=True)
        #------------------------------------------------
        fig.tight_layout()

        ax_pos = ax.get_position().bounds
        a1.set_position([ax_pos[0]+0.055, ax_pos[1]+ax_pos[3]-0.29, 0.25,0.25])
        a2.set_position([ax_pos[0]+ax_pos[2]-0.29, ax_pos[1]+0.055, 0.25, 0.25])



        if fig is None: return fig, ax
        else : return ax

    def plot_experiment_old(self, sujet=[0], mode_bcp=None, tau=40, direction=True, p=None, num_block=None, mode=None,
                        fig=None, axs=None, fig_width=15, titre=None, t_titre=35, t_label=25, return_proba=None,
                        color=[['k', 'k'], ['r', 'r'], ['k','w']], color_bet='r', color_va='k',
                        color_bcp='darkgreen', name_bcp='$P_{BBCP}$', print_suj=False, scaling_va=False,
                        alpha = [[.35,.15],[.35,.15],[1,0]], lw = 1.3, legends=False, TD=False, pause=50, ec = 0.2):

        import matplotlib.pyplot as plt
        import bayesianchangepoint as bcp
        from scipy import stats
        N_trials = self.param_exp['N_trials']
        N_blocks = self.param_exp['N_blocks']
        h = 1./tau
        if mode_bcp=='leaky' :
            fixed_window_size=tau
        else :
            fixed_window_size = 40


        if p is None : p = self.param_exp['p']
        if num_block is None : BLOCK = range(N_blocks)
        else: ec, BLOCK = 0.1, num_block

        ncol_leg = 2
        def plot_result_bcp(ax1, mode, observation, time, n_trial, name_bcp,  name=True) :

            from scipy.stats import beta
            p_bar, r_bar, beliefs = bcp.inference(observation, h=h, p0=p0, r0=r0)
            p_hat, r_hat = bcp.readout(p_bar, r_bar, beliefs, mode=mode, fixed_window_size=fixed_window_size, p0=p0)
            p_low, p_sup = np.zeros_like(p_hat), np.zeros_like(p_hat)

            for i_trial in range(n_trial):
                p_low[i_trial], p_sup[i_trial] = beta.ppf([.05, .95], a=p_hat[i_trial]*r_hat[i_trial], b=(1-p_hat[i_trial])*r_hat[i_trial])
            ax1.plot(time, p_hat, c=color_bcp, lw=1.5, alpha=.9, label=name_bcp if name is True else '')
            ax1.plot(time, p_sup, c=color_bcp, lw=1.2, alpha=.9, ls='--')#, label='CI' if name is True else '')
            ax1.plot(time, p_low, c=color_bcp, lw=1.2, alpha=.9, ls='--')
            ax1.fill_between(time, p_sup, p_low, lw=.5, alpha=.2, facecolor=color_bcp)

            return ax1


        if fig is None:
            mini_TD = False
            if len(sujet)==1 :
                if TD is True : mini_TD=True ; nb_ax=2
                else : fig, axs = plt.subplots(3, 1, figsize=(fig_width, fig_width/1.6180))

            else :
                if direction is True :
                    if TD is True : mini_TD=True ; nb_ax=len(sujet)
                    else : fig, axs = plt.subplots(len(sujet)+1, 1, figsize=(fig_width, ((len(sujet)+1)*fig_width/3)/(1.6180)))
                else :
                    fig, axs = plt.subplots(len(sujet), 1, figsize=(fig_width, ((len(sujet)+1)*fig_width/3)/(1.6180)))

            if mini_TD is True :
                import matplotlib.gridspec as gridspec
                #------------------------------------------------
                fig, axs = plt.subplots(nb_ax+1, 1, figsize=(fig_width, ((nb_ax+0.5)*fig_width/3)/(1.6180)))

                gs1 = gridspec.GridSpec(1, 1)
                gs1.update(left=0+0.072, bottom=0.85, right=1-0.04, top=1.-0.1, hspace=0.05)
                axs[0] = plt.subplot(gs1[0])

                if len(BLOCK)==1 : axs[0].plot(np.arange(1, N_trials), p[1:, BLOCK[0], 0], 'k.', ms=4)
                for card in ['bottom', 'top', 'right']: axs[0].spines[card].set_visible(False)
                axs[0].spines['left'].set_bounds(0, 1)

                gs2 = gridspec.GridSpec(nb_ax, 1)
                gs2.update(left=0+0.072, bottom=0+0.1, right=1-0.04, top=0.85-0.03, hspace=0.05)
                for s in range(nb_ax): axs[s+1] = plt.subplot(gs2[s])

        for i_layer in range(len(axs)):
            #------------------------------------------------
            # Barre Pause
            #------------------------------------------------
            if pause is not None :
                if pause > 0:
                    for num_pause in range(1,4) : axs[i_layer].bar(num_pause*pause-1, len(BLOCK)+ec*len(BLOCK), bottom=-ec/2, color='k', width=.5, linewidth=0)

            if num_block is None :
                #------------------------------------------------
                # affiche les numéro des block sur le côté gauche
                #------------------------------------------------
                ax_block = axs[i_layer].twinx()
                if i_layer==0 :
                    ax_block.set_ylabel('Block', fontsize=t_label/1.5, rotation='horizontal', ha='left', va='bottom')
                    ax_block.yaxis.set_label_coords(1.01, 1.08)

                ax_block.set_ylim(-.05, N_blocks + .05)
                ax_block.set_yticks(np.arange(N_blocks)+0.5)
                ax_block.set_yticklabels(np.arange(N_blocks)+1, fontsize=t_label/1.8)
                ax_block.yaxis.set_tick_params(width=0, pad=(t_label/1.5)+10)

            #------------------------------------------------
            # cosmétique
            #------------------------------------------------
            axs[i_layer].set_ylim(-(ec/2), len(BLOCK) +ec*len(BLOCK)-(ec/2))
            y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
            axs[i_layer].set_yticks(y_ticks[:len(BLOCK)*2])
            axs[i_layer].yaxis.set_label_coords(-0.05, 0.5)
            axs[i_layer].yaxis.set_tick_params(direction='out')
            axs[i_layer].yaxis.set_ticks_position('right')

            axs[i_layer].set_xlim(-1, N_trials)
            if i_layer==(len(axs)-1) :
                axs[i_layer].set_xticks([0, 50, 100, 150, 200])
                axs[i_layer].set_xticklabels([0, 50, 100, 150, 200], ha='left', fontsize=t_label/1.8)
                axs[i_layer].xaxis.set_ticks_position('bottom')
            else :
                axs[i_layer].set_xticks([])
        axs[0].yaxis.set_ticks_position('left')
        #------------------------------------------------
        # cosmétique
        #------------------------------------------------
        if len(sujet)==1 :
            axs[0].set_yticklabels(['left','right']*len(BLOCK),fontsize=t_label/1.8)

            y_ticks=[0, 0.5, 1, 1+ec, 1.5+ec, 2+ec, 2+ec*2, 2.5+ec*2, 3+ec*2]
            axs[1].set_yticks(y_ticks[:len(BLOCK)*3])
            axs[1].set_yticklabels(['0', '0.5', '1']*len(BLOCK),fontsize=t_label/1.8)
            axs[2].set_yticklabels(['No','Yes']*len(BLOCK),fontsize=t_label/1.8)
        else :
            if direction is True :
                y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
                axs[0].set_yticks(y_ticks[:len(BLOCK)*2])
                axs[0].set_yticklabels(['left','right']*len(BLOCK),fontsize=t_label/1.8)
            #else :
            #    axs[1].legend(fontsize=t_label/1.3, bbox_to_anchor=(0., 2.1, 1, 0.), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ###################################################################################################################################

        if TD is True : td_label = 'TD'
        else:           td_label = 'Target Direction'
        for i_block, block in enumerate(BLOCK):
            if len(sujet)==1 :
                for i_layer, label in enumerate([td_label, 'Probability', 'Switch']) :
                    if label == 'Switch' : axs[i_layer].step(range(N_trials), p[:, block, i_layer]+i_block+ec*i_block, lw=1, c=color[i_layer][0], alpha=alpha[i_layer][0])
                    axs[i_layer].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, i_layer])+ec*i_block, i_block+p[:, block, i_layer]+ec*i_block,
                                              lw=.5, alpha=alpha[i_layer][0], facecolor=color[i_layer][0], step='pre')
                    axs[i_layer].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, i_layer])+ec*i_block, i_block+p[:, block, i_layer]+ec*i_block,
                                              lw=.5, alpha=alpha[i_layer][1], facecolor=color[i_layer][1], step='pre')

                    axs[i_layer].set_ylabel(label, fontsize=t_label/1.2)
                    if mode=='deux' :
                        axs[1].text(-0.055, 0.5, 'Probability', fontsize=t_label, rotation=90, transform=axs[1].transAxes, ha='right', va='center')
            else :
                if direction is True :
                    axs[0].step(range(N_trials), p[:, block, 0]+i_block+ec*i_block, lw=1, c=color[0][0], alpha=alpha[0][0], where='mid')
                    axs[0].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 0])+ec*i_block,
                                              i_block+p[:, block, 0]+ec*i_block,
                                              lw=.5, alpha=alpha[0][0], facecolor=color[0][0], step='mid')
                    axs[0].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 0])+ec*i_block,
                                              i_block+p[:, block, 0]+ec*i_block,
                                              lw=.5, alpha=alpha[0][1], facecolor=color[0][1], step='mid')
                    axs[0].set_ylabel(td_label, fontsize=t_label/1.2)
                for s in range(len(sujet)) :
                    if direction is True : a = s+1
                    else : a = s
                    axs[a].step(range(N_trials), p[:, block, 1]+i_block+ec*i_block, lw=lw, c=color[1][0], alpha=1, label='$P_{true}$')
                    axs[a].fill_between(range(N_trials), i_block+np.zeros_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                              lw=.5, alpha=alpha[1][0], facecolor=color[1][0], step='pre')
                    axs[a].fill_between(range(N_trials), i_block+np.ones_like(p[:, block, 1])+ec*i_block, i_block+p[:, block, 1]+ec*i_block,
                                              lw=.5, alpha=alpha[1][1], facecolor=color[1][1], step='pre')

                    axs[a].plot(range(N_trials), 0.5*np.ones(N_trials)+i_block+ec*i_block, lw=1.5, c='k', alpha=0.5)
                    axs[a].text(-0.055, 0.5, 'Subject %s'%(s), fontsize=t_label/1.2, rotation=90, transform=axs[a].transAxes, ha='right', va='center')
        #-------------------------------------------------------------------------------------------------------------

        for s in range(len(sujet)) :
            if direction is True : a = s+1
            else :                 a = s

            if len(sujet)==1: y_t = 1.1
            else :            y_t = 1.25
            suj = sujet[s]
            p = self.BET[self.subjects[suj]]['p']
            results = (self.BET[self.subjects[suj]]['results']+1)/2 # results est sur [-1,1] on le ramene sur [0,1]
            a_anti, start_anti, latency = self.EYEMVT[self.subjects[suj]]['a_anti'], self.EYEMVT[self.subjects[suj]]['start_anti'], self.EYEMVT[self.subjects[suj]]['latency']
            if print_suj is True : print('sujet', suj, '=', self.subjects[suj])

            #-------------------------------------------------------------------------------------------------------------

            if scaling_va is True : mini=0.5
            else :                  mini = 5 #8
            ec1 = ec*mini*2

            if titre is None :
                if mode == 'bet' :             axs[0].set_title('Bet results', fontsize=t_titre, x=0.5, y=y_t)
                elif mode == 'eyeMvt' : axs[0].set_title('Eye movements recording results', fontsize=t_titre, x=0.5, y=y_t)
                elif mode=='deux':              axs[0].set_title('Bet + Eye movements results', fontsize=t_titre, x=0.5, y=y_t)

            if mode in ['bet', 'deux'] :

                for i_block, block in enumerate(BLOCK):
                    axs[a].step(range(N_trials), i_block+results[:, block]+ec*i_block, lw=lw, alpha=1, color=color_bet, label='Individual guess'  if i_block==0 else '')

                axs[a].yaxis.set_ticks_position('left')
                y_ticks=[0, 0.5, 1, 1+ec, 1.5+ec, 2+ec, 2+ec*2, 2.5+ec*2, 3+ec*2]
                axs[a].set_yticks(y_ticks[:len(BLOCK)*3])
                axs[a].set_yticklabels(['0', '0.5', '1']*len(BLOCK),fontsize=t_label/1.8)

                axs[a].set_ylabel('Bet score', fontsize=t_label/1.5, color=color_bet)
                axs[a].tick_params('y', colors=color_bet)
                axs[a].yaxis.set_label_coords(-0.03, 0.5)

            if mode in ['eyeMvt', 'deux'] :

                ax1 = axs[a].twinx()
                for i_block, block in enumerate(BLOCK):
                    axs[a].step(range(1), -1000, color=color_va, lw=lw, alpha=1, label='Eye movement'  if i_block==0 else '')
                    va = (np.array(a_anti[block])*((np.array(latency[block])-np.array(start_anti[block]))/1000))
                    if scaling_va is True : va=(va-np.min(va))/(np.max(va)-np.min(va))

                    ax1.step(range(N_trials), 2*(mini*i_block)+va+ec1*i_block,
                                color=color_va, lw=lw, alpha=1, label='Eye movement' if i_block==0 else '')



                if scaling_va is True :
                    y_ticks=[0, 1, 1+ec, 2+ec, 2+ec*2, 3+ec*2]
                    ax1.set_yticklabels(['min', 'max']*len(BLOCK), fontsize=t_label/1.8)
                    ax1.set_yticks(y_ticks[:len(BLOCK)*3])
                    ax1.yaxis.set_label_coords(1.06, 0.5)


                else:
                    ax1.set_ylim(-mini-(ec1/2), len(BLOCK)*mini + ec1*len(BLOCK)-(ec1/2))
                    y_ticks=[-mini, 0, mini,
                             mini+ec1, 2*mini+ec1, 3*mini+ec1,
                             3*mini+2*ec1, 4*mini+2*ec1, 5*mini+2*ec1]

                    ax1.set_yticks(y_ticks[:len(BLOCK)*3])
                    ax1.set_yticklabels(['-%s'%mini, '0', '%s'%mini]*len(BLOCK),fontsize=t_label/1.8)
                    ax1.yaxis.set_label_coords(1.043, 0.5)
                ax1.yaxis.set_tick_params(colors=color_va, direction='out')
                ax1.yaxis.set_ticks_position('right')
                #ax1.set_ylabel('Anticipatory eye\nvelocity °/s', rotation=-90,fontsize=t_label/1.5)
                #ax1.set_ylabel('Velocity of anticipation °/s', rotation=-90,fontsize=t_label/1.5)

                ax1.set_ylabel('Velocity of eye °/s', rotation=-90, fontsize=t_label/1.5, color=color_va)
                #if mode == 'eyeMvt' : axs[a].set_yticks([])


            if mode_bcp is not None :
                ncol_leg = 4
                p0, r0 =  0.5, 1.0
                p = self.BET[self.subjects[s]]['p']
                if pause is not None :
                    liste = [0,50,100,150,200]
                    for pause in range(len(liste)-1) :
                        if pause==0 and s==0 : name=True
                        else :                 name=False
                        n_trial = liste[pause+1]-liste[pause]
                        axs[a] = plot_result_bcp(axs[a], mode_bcp, p[liste[pause]:liste[pause+1], block, 0],
                                                 np.arange(liste[pause], liste[pause+1]), n_trial, name_bcp, name=name)
                else :
                    if s==0 : name=True
                    else :    name=False
                    axs[a] = plot_result_bcp(axs[a], mode_bcp, p[:, block, 0], np.arange(N_trials), N_trials, name_bcp, name=name)

            #------------------------------------------------
            if mode is None and titre is None : axs[0].set_title('Experiment', fontsize=t_titre, x=0.5, y=y_t)
            #-------------------------------------------------------------------------------------------------------------

            if titre is not None : axs[0].set_title(titre, fontsize=t_titre, x=0.5, y=y_t)


        if legends is True :
            if TD is True : axs[1].legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 1.3, 1, 0.), loc=3, ncol=ncol_leg, mode="expand", borderaxespad=0.)
            else :          axs[1].legend(fontsize=t_label/1.8, bbox_to_anchor=(0., 2.1, 1, 0.), loc=3, ncol=ncol_leg, mode="expand", borderaxespad=0.)

        axs[-1].set_xlabel('Trial #', fontsize=t_label)
        try: fig.tight_layout()
        except: print('tight_layout failed :-(')
        plt.subplots_adjust(hspace=0.05)
        #------------------------------------------------

        if return_proba is None : return fig, axs
        else : return fig, axs, p


if __name__ == '__main__':

    try:
        mode = sys.argv[1]
    except:
        mode = 'bet'
        #mode = 'eyeMvt'
    try:
        timeStr = sys.argv[4]
    except:
        import time
        timeStr = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
        #timeStr = '2017-06-22_102207'

    e = aSPEM(mode, timeStr)
    if not mode is 'model':
        print('Starting protocol')
        e.run_experiment()
