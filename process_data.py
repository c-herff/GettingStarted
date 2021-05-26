import numpy as np 
import numpy.matlib as matlib
from mne.filter import filter_data
import scipy.io.wavfile as wav
import scipy
from scipy import fftpack
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode
import MelFilterBank as mel
from scipy.signal import decimate, hilbert

# Helper function to drastically speed up the hilbert transform of larger data
hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

def extractHG(data, sr,windowLength=0.05,frameshift=0.01):
    """Window data and extract high-gamma envelope using the hilbert transform
    
    Parameters
    ----------
    data: array (samples, channels)
        EEG time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    feat, array shape (windows, channels)
        High-gamma feature matrix
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    numWindows=int(np.floor((data.shape[0]-windowLength*sr)/(frameshift*sr)))
    # Band-pass for high-gamma between 70 and 170 Hz)
    data = filter_data(data.T, sr, 70,170,method='iir').T
    # Band-stop filter for first two harmonics of 50 Hz line noise
    data = filter_data(data.T, sr, 102, 98,method='iir').T # Band-stop
    data = filter_data(data.T, sr, 152, 148,method='iir').T
    data = np.abs(hilbert3(data))
    feat = np.zeros((numWindows,data.shape[1]))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        feat[win,:] = np.mean(data[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5):
    """Add temporal context to each window by stacking neighboring feature vectors
    
    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    Returns
    ----------
    featStacked, array shape (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    featStacked=np.zeros((features.shape[0]-(2*modelOrder*stepSize),(2*modelOrder+1)*features.shape[1]))
    for fNum,i in enumerate(range(modelOrder*stepSize,features.shape[0]-modelOrder*stepSize)):
        ef=features[i-modelOrder*stepSize:i+modelOrder*stepSize+1:stepSize,:]
        featStacked[fNum,:]=ef.flatten() # Add 'F' if stacked the same as matlab
    return featStacked

def downsampleLabels(labels, sr, windowLength=0.05, frameshift=0.01):
    """Downsamples non-numerical data by using the mode
    
    Parameters
    ----------
    labels: array of str
        Label time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels array of str
        downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-windowLength*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+windowLength*sr))
        newLabels[w]=mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def windowAudio(audio, sr, windowLength=0.05, frameshift=0.01):
    """Window Audio data
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the data
    windowLength: float
        Length of window (in seconds) for which raw audio will be extracted
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    winAudio array (windows, audiosamples)
        Windowed audio
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    winAudio = np.zeros((numWindows, int(windowLength*sr )))
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))    
        winAudio[w,:] = audio[startAudio:stopAudio]
    return winAudio

#
def extractMelSpecs(audio, sr, windowLength=0.05, frameshift=0.01,numFilter=23):
    """Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms
    
    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram, array shape (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-windowLength*sr)/(frameshift*sr)))
    win = scipy.hanning(np.floor(windowLength*sr + 1))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(windowLength*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        startAudio = int(np.floor((w*frameshift)*sr))
        stopAudio = int(np.floor(startAudio+windowLength*sr))
        a = audio[startAudio:stopAudio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], numFilter, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs,modelOrder=4):
    """Creates list of electrode names.
    
    Parameters
    ----------
    elecs: array of strings 
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names

    Returns
    ----------
    names: array of strings 
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  # Add 'F' if stacked the same as matlab


if __name__=="__main__":
    winL = 0.05 # 0.01
    frameshift = 0.01 #0.01
    modelOrder=4
    stepSize=5
    path = r'./'
    outPath = r'./'
    pts = ['kh9']
    sessions = [1]
    
    for pNr, p in enumerate(pts):
        for ses in range(1,sessions[pNr]+1):
            dat = np.load(path + '/' + p + '_' + str(ses)  + '_sEEG.npy')
            sr=1024
            # Extract High-Gamma features
            feat = extractHG(dat,sr, windowLength=winL,frameshift=frameshift)

            
            
            
            # Extract labels
            words=np.load(path + '/' + p + '_' + str(ses) + '_words.npy')
            words=downsampleLabels(words,sr,windowLength=winL,frameshift=frameshift)
            words=words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]

            # Load audio
            audio = np.load(path + '/' + p + '_' + str(ses)  + '_audio.npy')
            audioSamplingRate = 48000
            # Downsample Audio to 16kHz
            targetSR = 16000
            audio = decimate(audio,int(audioSamplingRate / targetSR))
            audioSamplingRate = targetSR
            # Write wav file of the audio
            scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
            wav.write(outPath + p + '_' + str(ses)  + '_orig_audio.wav',audioSamplingRate,scaled)   
            # Extact log mel-scaled spectrograms
            melSpec = extractMelSpecs(scaled,audioSamplingRate,windowLength=winL,frameshift=frameshift,numFilter=23)
            # Raw audio aligned to each window (for unit selection)
            winAudio = windowAudio(scaled, audioSamplingRate,windowLength=winL,frameshift=frameshift)
            
            # Uncomment if you want to use feature stacking
            ## Stack features
            #feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)
            ## Align to EEEG features
            #melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
            #winAudio = winAudio[modelOrder*stepSize:winAudio.shape[0]-modelOrder*stepSize,:]
            if melSpec.shape[0]!=feat.shape[0]:
                print('Possible Problem with ECoG/Audio alignment for %s session %d.' % (p,ses))
                print('Diff is %d' % (np.abs(feat.shape[0]-melSpec.shape[0])))
                tLen = np.min([melSpec.shape[0],feat.shape[0]])
                melSpec = melSpec[:tLen,:]
                winAudio = winAudio[:tLen,:]
                feat = feat[:tLen,:]
            # Save everything
            np.save(outPath + p + '_' + str(ses)  + '_feat.npy', feat)
            np.save(outPath + p + '_' + str(ses)  + '_procWords.npy', words)
            np.save(outPath + p + '_' + str(ses)  + '_spec.npy',melSpec)
            np.save(outPath + p + '_' + str(ses)  + '_winAudio.npy',winAudio)
            
            elecs = np.load(path + p + '_' + str(ses)  + '_channelNames.npy')
            ##Add context 
            #elecs = nameVector(elecs, modelOrder=modelOrder)
            np.save(outPath + p + '_' + str(ses)  + '_feat_names.npy', elecs)