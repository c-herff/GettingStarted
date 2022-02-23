import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score


if __name__=="__main__":
    path = r'./'
    outPath = r'./'
    pts = ['kh9']
    sessions = [1]
    nfolds = 10

    allResults = np.zeros((len(pts),nfolds))
    for pNr, patient in enumerate(pts):
        for ses in range(1,sessions[pNr]+1):

            # Load preprocessed data
            features = np.load(path + patient + '_' + str(ses)  + '_feat.npy')
            spectrogram = np.load(path + patient + '_' + str(ses)  + '_spec.npy')

            #From the spectrogram we simply determine whether there is speech or not
            #This could be changed to any kind of label
            
            label = np.mean(spectrogram,axis=1)>5

            #We do 10 fold cross-validation
            #That means we train on 90% of the data and test on the remaining 10 percent
            #We repeat this 10 times until all data was used for testing exactly ones
            
            kf = KFold(nfolds,shuffle=False)
            
            est = LinearDiscriminantAnalysis()
            for k,(train, test) in enumerate(kf.split(features)):
                #Fit the classifier on the training data of this fold
                est.fit(features[train,:], label[train])
                #Get probabilities for each class on the testing data
                predicted_probs = est.predict_proba(features[test,:])
                #Evaluate results for this fold:
                score = roc_auc_score(label[test],predicted_probs[:,1])
                allResults[pNr,k]=score
    # Classification is done at this point
    print('Mean AUC is %f' % np.mean(allResults))