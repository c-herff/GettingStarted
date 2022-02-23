# GettingStarted
Collection of scripts and notebooks to get you started with our sEEG data.

In this example, we use a single word speech experiment data set.

### load_data.py
1. Loads the xdf file
2. Aligns the different streams (eeg, audio, experiment markers)
3. Saves them into a numpy format

### inspectRawData.ipynb
1. Loads the numpy files
2. Plots raw data and different band-passed signals
3. Windows data and calculates the envelope
4. Calculates ERPs
5. Calculates Power spectrum
6. Calculates spectrograms

### process_data.py
1. Loads the numpy files
2. Calculates high gamma features
3. Calculates logarithmic mel-scaled spectrograms
4. Saves everything

### classification_example.py
1. Loads the features from process_data
2. Transforms spectrograms into binary labels (speech yes or no)
3. Trains and evaluates LDA on the features in a k-fold cross-validation

