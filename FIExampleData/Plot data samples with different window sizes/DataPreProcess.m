load ('tremorEpoch.mat')
load('tremorTimestamp.mat')
load('tremorValue.mat')
load ('activityEpoch.mat')
load('activityTimestamp.mat')
load('activityValue.mat')
load('medicineEpoch')



windowSize = 10;   %assign the window size, measured by minutes
bedTime = 11;     %assume every day sleeps at a spcific time , measured in 
                  %hours PM, e.g. bedTime = 11, everyday sleeps at 11:00 pm

samplePlot( tremorValue,activityValue,windowSize,medicineEpoch,tremorEpoch,bedTime );