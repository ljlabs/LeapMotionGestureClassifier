# LeapMotionGestureClassifier

This is a toy implantation of a custom Leap Motion Hand  
gesture classifier.

It essentailly uses a 1D convelutional neural network and  
an LSTM to classify hand gestures into 1 of 3 different  
classes `['wiggle', 'grab', 'hand_flick']`

The model was trained using google colab, and I have added  
both the final trained model, and the saved network weights  
with the highest validation accuracy during training. 

For preprocessing, which is not yet handled in this repo  
`but may be added soon` I used the steps outlined in this   
paper [DOI: 10.3390/s20072106](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7180537/)