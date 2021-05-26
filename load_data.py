import pyxdf
import bisect
import numpy as np
from scipy.signal import decimate


## Find timestamps that are the best match
def locate_pos(available_ts, target_ts):
    """Finds best fitting time point.
    
    Parameters
    ----------
    available_ts: array of floats 
        Time stamps of time series
    target_ts: float
        Time stamp that should be found in list of time stamps

    Returns
    ----------
    pos: integer 
        index of time stamp with smallest distance from target_ts
    """
    pos = bisect.bisect_right(available_ts, target_ts)
    if pos == 0:
        return 0
    if pos == len(available_ts):
        return len(available_ts)-1
    if abs(available_ts[pos]-target_ts) < abs(available_ts[pos-1]-target_ts):
        return pos
    else:
        return pos-1  

if __name__=="__main__":
    pts = ['kh9']
    sessions = [1,]
    path = r'./'
    outPath = r'./'
    for pNr, p in enumerate(pts):
        for ses in range(1,sessions[pNr]+1):
            
            # Load xdf file
            streams = pyxdf.load_xdf(path + p + '/speech' + str(ses) + '.xdf',dejitter_timestamps=False)
            streamToPosMapping = {}
            for pos in range(0,len(streams[0])):
                stream = streams[0][pos]['info']['name']
                streamToPosMapping[stream[0]] = pos

            # Get sEEG data
            eeg = streams[0][streamToPosMapping['Micromed']]['time_series']
            offset = float(streams[0][streamToPosMapping['Micromed']]['info']['created_at'][0])
            # Get corresponding time stamps for each sample
            eeg_ts = streams[0][streamToPosMapping['Micromed']]['time_stamps'].astype('float')#+offset
            # Sampling rate
            eeg_sr = int(streams[0][streamToPosMapping['Micromed']]['info']['nominal_srate'][0])
            # Some data sets are sampled with 2kHz, in that case we downsample
            if eeg_sr == 2048:
                eeg = decimate(eeg,2,axis=0)
                eeg_ts = eeg_ts[::2]
            #Get electrode info
            chNames = []
            for ch in streams[0][streamToPosMapping['Micromed']]['info']['desc'][0]['channels'][0]['channel']:
                chNames.append(ch['label'])

            #Load Audio
            audio = streams[0][streamToPosMapping['AudioCaptureWin']]['time_series']
            offset_audio = float(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['created_at'][0])
            # Corresponding audio time stamps for each sample
            audio_ts = streams[0][streamToPosMapping['AudioCaptureWin']]['time_stamps'].astype('float')#+offset
            # Audio sampling rate
            audio_sr = int(streams[0][streamToPosMapping['AudioCaptureWin']]['info']['nominal_srate'][0]) 
            
            # Load Marker stream which contains the experiment timings
            markers = streams[0][streamToPosMapping['SingleWordsMarkerStream']]['time_series']
            offset_marker = float(streams[0][streamToPosMapping['SingleWordsMarkerStream']]['info']['created_at'][0])
            marker_ts = streams[0][streamToPosMapping['SingleWordsMarkerStream']]['time_stamps'].astype('float')#-offset

            #Process Experiment timing
            i=0
            while markers[i][0]!='experimentStarted':
                i+=1
            # Find time stamp in eeg timestamps that is closest to the experiment start marker
            eeg_start= locate_pos(eeg_ts, marker_ts[i])
            # Find time stamp in audio timestamps that is closest to the experiment start marker
            audio_start = locate_pos(audio_ts, eeg_ts[eeg_start])
            while markers[i][0]!='experimentEnded':
                i+=1
            #Find time tamp in eeg timestamps that is closest to the experiment end marker
            eeg_end= locate_pos(eeg_ts, marker_ts[i])
            #Find time tamp in audio timestamps that is closest to the experiment end marker
            audio_end = locate_pos(audio_ts, eeg_ts[eeg_end])
            markers=markers[:i]
            marker_ts=marker_ts[:i]

            # Cut out only the audio and eeg during the experiment
            eeg = eeg[eeg_start:eeg_end,:]
            eeg_ts = eeg_ts[eeg_start:eeg_end]
            audio = audio[audio_start:audio_end,:]
            audio_ts=audio_ts[audio_start:audio_end]

            # Initialize corresponding labels for the time series
            words=['' for a in range(eeg.shape[0])]
            #Get only the starts for each word
            wordMask = [m[0].split(';')[0]=='start' for m in markers]
            wordStarts = marker_ts[wordMask]
            #Find the corresponding eeg time stamps
            wordStarts = np.array([locate_pos(eeg_ts, x) for x in wordStarts])
            #Extract only the word from the marker
            dispWords =  [m[0].split(';')[1] for m in markers if m[0].split(';')[0]=='start']
            # Same procedure for the word ends
            wordEndMask = [m[0].split(';')[0]=='end' for m in markers]
            wordEnds = marker_ts[wordEndMask]
            wordEnds = np.array([locate_pos(eeg_ts, x) for x in wordEnds])

            # Set the label vector with the words
            for i, start in enumerate(wordStarts):
                words[start:wordEnds[i]]=[dispWords[i] for rep in range(wordEnds[i]-start)]
            print('All aligned')
            # Saving
            #Adding some white noise because the microphone thresholds the data
            noise = np.random.normal(0,0.0001, audio.shape[0])
            np.save(outPath  + p + '_' + str(ses) + '_sEEG.npy', eeg)
            np.save(outPath  + p + '_' + str(ses) +'_words.npy', np.array(words))
            np.save(outPath  + p + '_' + str(ses) +'_channelNames.npy', np.array(chNames))
            np.save(outPath  + p + '_' + str(ses) +'_audio.npy',audio[:,0]+noise)