function Y1=Dep_convent_network(b1,a,c)
load data1
load data2
T1=[data1,data2];
x=[0 1];
Y1=a;
Tl=b1;
net1=c;
net1 = complx_net_feat(minmax(T1),[20 10 1],{'logsig','logsig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.04;
net1.trainParam.epochs = 7000;
net1.trainParam.goal = 1e-5;[net1] = train(net1,T1,x);
save net1 net1
Yl = round(sim(net1,T1));
end
