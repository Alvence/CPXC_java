addpath(genpath('.'))
filename = 'data/synthetic2.train';
M = csvread(filename);

X=M(:,1,:)
Y=M(:,2,:)
figure
hold on
scatter(X,Y)

filename = 'data/synthetic2.test';
M = csvread(filename);

X=M(:,1,:)
Y=M(:,2,:)
scatter(X,Y)