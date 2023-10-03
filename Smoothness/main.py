import os

#import cv2
import numpy as np
#import moviepy.editor as mp
import matplotlib
import matplotlib.pyplot as plt

from pandas import *
import numpy as np


from matplotlib.figure import Figure


# Functions from https://github.com/siva82kb/SPARC
# Code permitted for modification under a ISC license
def sparc(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified
    spectral arc length metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]
    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.
    new_sal, (f, Mf), (f_sel, Mf_sel)
    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = sparc(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'
    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs / nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf / max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc) * 1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th) * 1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1] + 1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel) / (f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)


def dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    dimensionless jerk metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'
    """
    # first enforce data into an numpy array.
    movement = np.array(movement)

    # calculate the scale factor and jerk.
    movement_peak = max(abs(movement))
    dt = 1. / fs
    movement_dur = len(movement) * dt
    jerk = np.diff(movement, 2) / pow(dt, 2)
    scale = pow(movement_dur, 3) / pow(movement_peak, 2)

    # estimate dj
    return - scale * sum(pow(jerk, 2)) * dt


def log_dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the
    log dimensionless jerk metric.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk(movement, fs)))


def dimensionless_jerk2(movement, fs, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'
    """
    # first ensure the movement type is valid.
    if data_type in ('speed', 'accl', 'jerk'):
        # first enforce data into an numpy array.
        movement = np.array(movement)

        # calculate the scale factor and jerk.
        movement_peak = max(abs(movement))
        dt = 1. / fs
        movement_dur = len(movement) * dt
        # get scaling factor:
        _p = {'speed': 3,
              'accl': 1,
              'jerk': -1}
        p = _p[data_type]
        scale = pow(movement_dur, p) / pow(movement_peak, 2)

        # estimate jerk
        if data_type == 'speed':
            jerk = np.diff(movement, 2) / pow(dt, 2)
        elif data_type == 'accl':
            jerk = np.diff(movement, 1) / pow(dt, 1)
        else:
            jerk = movement

        # estimate dj
        return - scale * sum(pow(jerk, 2)) * dt
    else:
        raise ValueError('\n'.join(("The argument data_type must be either",
                                    "'speed', 'accl' or 'jerk'.")))


def log_dimensionless_jerk2(movement, fs, data_type='speed'):
    """
    Calculates the smoothness metric for the given movement data using the
    log dimensionless jerk metric. The input movement data can be 'speed',
    'accleration' or 'jerk'.
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    data_type: string
               The type of movement data provided. This will determine the
               scaling factor to be used. There are only three possibiliies,
               {'speed', 'accl', 'jerk'}
    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's
               smoothness.
    Notes
    -----
    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'
    """
    return -np.log(abs(dimensionless_jerk2(movement, fs, data_type)))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



if __name__ == '__main__':
    # Spectral Arc Length Calculations
    # files=["data_20220407_111841_216.csv","data_20220407_111910_715.csv","data_20220407_111957_691.csv"]
    files=["High-Aim-Smooth.csv","High-Aim-Jerky.csv","High-Aim-Jerkiest.csv"]
    sparc_output_x_COLLEC=[]
    sparc_output_y_COLLEC = []
    SPARC_OUTPUT_QUART_Y_SINGLES_COLLEC=[]
    LDJ_OUTPUT_QUART_Y_SINGLES_COLLEC=[]
    sparc_output_z_COLLEC = []
    SPARC_OUTPUT_QUART_Y_COLLEC=[]
    LDJ_OUTPUT_QUART_Y_COLLEC=[]
    TIME_SERIES_COLLEC=[]
    for file in files:
        print('Performing analysis on file:')
        print(file)
        data = read_csv(file)
        GYR_X = data['FreeAcc_X'].tolist()
        GYR_Y = data['FreeAcc_Y'].tolist()
        GYR_Z = data['FreeAcc_Z'].tolist()
        QUART_Y_1=data['Quat_Y'].tolist()

        sparc_output_x = sparc(np.array(GYR_X), 60.0, 4, 100.0, 0.05)

        sparc_output_x_COLLEC.append(sparc_output_x)
        sparc_output_y = sparc(np.array(GYR_Y), 60.0, 4, 100.0, 0.05)
        sparc_output_x_COLLEC.append(sparc_output_x)
        sparc_output_z = sparc(np.array(GYR_Z), 60.0, 4, 100.0, 0.05)
        sparc_output_quart_y = sparc(np.array(QUART_Y_1), 60.0, 4, 100.0, 0.05)
        SPARC_OUTPUT_QUART_Y_COLLEC.append(sparc_output_quart_y)
        print('SPARC Output For FreeAcc_X')
        print(sparc_output_x)
        print('SPARC Output For FreeAcc_Y')
        print(sparc_output_y)
        print('SPARC Output For FreeAcc_Z')
        print(sparc_output_z)
        log_dimless_jerk_x = dimensionless_jerk2(GYR_X, 60.0, 'accl')
        log_dimless_jerk_y = dimensionless_jerk2(GYR_Y, 60.0, 'accl')
        log_dimless_jerk_z = dimensionless_jerk2(GYR_Z, 60.0, 'accl')
        print('Log Dimensionless Jerk X:')
        print(log_dimless_jerk_x)
        print('Log Dimensionless Jerk Y:')
        print(log_dimless_jerk_y)
        print('Log Dimensionless Jerk Z:')
        print(log_dimless_jerk_z)

        # Chunk size of 3 correlates to 20 datapoints per second
        GYR_X_blocks = list(chunks(GYR_X, 3))
        GYR_Y_blocks = list(chunks(GYR_Y, 3))
        GYR_Z_blocks = list(chunks(GYR_Z, 3))
        QUART_Y_BLOCKS=list(chunks(QUART_Y_1,3))

        averaged_GYR_X = []
        LDJ_QUART_Y_BLOCKS=[]
        LDJ_X_blocks = []
        SPARC_X_values = []
        SPARC_X_blocks = []
        LDJ_Y_blocks = []
        LDJ_Z_blocks = []
        time_series = []
        time_val = 0
        gyr_av_sum = 0
        average = 0
        counter = 0
        for x in GYR_X:
            counter += 1
            if counter % 3 == 0:
                average = gyr_av_sum / 3.0
                averaged_GYR_X.append(average)
                gyr_av_sum = 0
            else:
                gyr_av_sum = gyr_av_sum + x

        print(averaged_GYR_X)
        for x in QUART_Y_BLOCKS:
            LDJ_QUART_Y_BLOCKS.append(log_dimensionless_jerk(x, 60.0))
        for x in GYR_X_blocks:
            time_val += 0.05
            time_series.append(time_val)
            LDJ_X_blocks.append(log_dimensionless_jerk(x, 60.0))
        for x in GYR_Y_blocks:
            LDJ_Y_blocks.append(log_dimensionless_jerk(x, 60.0))
        for x in GYR_Z_blocks:
            LDJ_Z_blocks.append(log_dimensionless_jerk(x, 60.0))
        for x in GYR_X_blocks:
            SPARC_X_blocks.append(sparc(x, 40.0, 4, 2.0, 0.05))

        for x in SPARC_X_blocks:
            SPARC_X_values.append(x[0])
            """
            sal      : float
                       The spectral arc length estimate of the given movement's
                       smoothness.
            (f, Mf)  : tuple of two np.arrays
                       This is the frequency(f) and the magntiude spectrum(Mf) of the
                       given movement data. This spectral is from 0. to fs/2.
            (f_sel, Mf_sel) : tuple of two np.arrays
                              This is the portion of the spectrum that is selected for
                              calculating the spectral arc length.
            new_sal, (f, Mf), (f_sel, Mf_sel)
            """
        LDJ_OUTPUT_QUART_Y_COLLEC.append(LDJ_QUART_Y_BLOCKS)
        TIME_SERIES_COLLEC.append(time_series)
        #print('Log dimensionless jerk values, with a window of 100ms')
        #print(LDJ_X_blocks)

        #plt.plot(time_series, LDJ_X_blocks, label='X Jerk')
        #plt.plot(time_series, LDJ_Y_blocks, label='Y Jerk')
        #plt.plot(time_series, LDJ_Z_blocks, label='Z Jerk')
        #plt.figure()
        #plt.legend()
        #plt.xlabel('Time (s)')
        #plt.ylabel('LDJ Steadiness Value')
        #plt.show()
        
        
        
        plt.plot(time_series, LDJ_X_blocks, label='X Jerk-LDJ')
        plt.plot(time_series, SPARC_X_values, label='X Jerk-SPARC')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('LDJ Steadiness Value')
        plt.title(file)
        plt.show()
        #plt.legend()


    #plt.plot(time_series, LDJ_QUART_Y_BLOCKS, label='Y QUART Jerk')
    ##plt.plot(time_series, LDJ_Y_blocks, label='Y Jerk')
    ##plt.plot(time_series, LDJ_Z_blocks, label='Z Jerk')
    #plt.xlabel('Time (s)')
    #plt.ylabel('LDJ Steadiness Value')
    #plt.show()

    #plt.figure(1)
    #plt.plot(time_series, LDJ_X_blocks, label='X Jerk-LDJ')
    #plt.plot(time_series, SPARC_X_values, label='X Jerk-SPARC')

    #plt.legend()
    #plt.xlabel('Time (s)')
    #plt.ylabel('')
    #plt.show()


    #plt.figure(2)
    #plt.plot(TIME_SERIES_COLLEC[1], LDJ_OUTPUT_QUART_Y_COLLEC[1], label='X Jerk-SPARC 2')
    #plt.legend()
    #plt.xlabel('Time (s)')
    #plt.ylabel('')
    #plt.show()

        #time_val += 0.05
        #time_series.append(time_val)
        #SPARC_X_blocks_2 = []
        #time_series_2 = []
        #time_val_2 = 0

    # fp = "temp_resized.mp4"
    # clip = mp.VideoFileClip("DSC_1059.MOV")
    # clip_resized = clip.resize(height=(680), width=(700))  # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    # clip_resized.write_videofile("temp_resized.mp4")
    # app = wx.App(0)
    # myframe = MyFrame(fp)
    # app.MainLoop()