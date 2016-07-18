function samplePlot( tremorValue,activityValue,windowSize,medicineEpoch,tremorEpoch,bedTime )
%This is a function to plot the sample for the signal1 and signal2. The
%sample rate is defined in sampleTime. Specifically the sampleTime should
%be multiples of minutes


%% convert the medicine epochs to window size scales
relativeMedicineEpochMinute = floor((medicineEpoch - tremorEpoch(1))/1000/60);
MedicineEpochInWindows = relativeMedicineEpochMinute/windowSize;


%% The preprocess initializations on data,
% Basically it is about use the median of data in every minute as samples


%combine how many points for plots, tremorValueInOneWindow means every point
%is the median of the tremor score within that window. 
%The same goes for activityValueInOneWindow.
tremorValueInOneWindow = 12*windowSize; 
activityValueInOneWindow = 2*windowSize;

time_start = 5; %starting time is 5 pm 
windowSizePerDay = 24*60/windowSize;
sleepTime = [0:3] * windowSizePerDay + (bedTime-time_start)*60/windowSize;


%% tremorSample is the median of the values in each window
numOfTremorValue = length(tremorValue);

tremorValueCutTail = tremorValue(1    :  floor(numOfTremorValue/tremorValueInOneWindow)*tremorValueInOneWindow);
%tremorValueCutTail denotes tremorValue droping the last several points
%less than a window size
tremorSample = median(reshape(tremorValueCutTail, tremorValueInOneWindow,length(tremorValueCutTail)/tremorValueInOneWindow),1)';

%% plot sampling of tremor scores
figure(1)
plot(1:floor(numOfTremorValue/tremorValueInOneWindow),tremorSample);
str = sprintf('Tremor scores samples\nWindow size is %d minite(s)',windowSize);
title(str);
% add markers for the timepoints of taking medicine
h1 = animatedline('Marker','*','Color','b','LineWidth',0.01);
x1 = MedicineEpochInWindows;
y1 = max(tremorSample)*ones(length(MedicineEpochInWindows),1);
h2 = animatedline('Marker','o','Color','r','LineWidth',0.01);
x2 = sleepTime;
y2 = 0*ones(length(sleepTime),1);
addpoints(h1,x1,y1);
addpoints(h2,x2,y2);


%% activitySample is the median of the values in each window
numOfActivityValue = length(activityValue);

activityValueCutTail = activityValue(1    :  floor(numOfActivityValue/activityValueInOneWindow)*activityValueInOneWindow);
%activityValueCutTail denotes activityValue droping the last several points
%less than a window size

activitySample = median(reshape(activityValueCutTail, activityValueInOneWindow,length(activityValueCutTail)/activityValueInOneWindow),1)';



%% plot sampling of activity levels
figure(2)
plot(1:floor(numOfActivityValue/activityValueInOneWindow),activitySample);
str = sprintf('Activity level samples\nWindow size is %d minite(s)',windowSize);
title(str);

h1 = animatedline('Marker','*','Color','b','LineWidth',0.01);
y1 = max(activitySample)*ones(length(MedicineEpochInWindows),1);
h2 = animatedline('Marker','o','Color','r','LineWidth',0.01);
addpoints(h1,x1,y1);
addpoints(h2,x2,y2);
end

