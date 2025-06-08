function [Iout1,intensity,fitness,time]=fa_edge(I,level,method)
tic;
if (nargin<2)   %didn't choose level and methods
    level=3;
    method='fa';
end
if (nargin<3)   %didn't choose methods
    method='fa';
end

if size(I,3)==1 %grayscale image
    [n_countR,x_valueR] = imhist(I(:,:,1));
elseif size(I,3)==3 %RGB image
    [n_countR,x_valueR] = imhist(I(:,:,1));
    [n_countG,x_valueG] = imhist(I(:,:,2));
    [n_countB,x_valueB] = imhist(I(:,:,3));
end

Nt=size(I,1)*size(I,2);
Lmax=256;   %256 different maximum levels are considered in an image (i.e., 0 to 255)

for i=1:Lmax
    if size(I,3)==1 %grayscale image
        probR(i)=n_countR(i)/Nt;
    elseif size(I,3)==3 %RGB image    
        probR(i)=n_countR(i)/Nt;
        probG(i)=n_countG(i)/Nt;
        probB(i)=n_countB(i)/Nt;
    end
end

if strcmpi(method,'fa') %fa method

    N = 150; %predefined fa population for multi-segmentation

    N_PAR = level-1;  %number of thresholds (number of levels-1)

    N_GER = 15; %number of iterations of the fa algorithm

    PHI1 = 0.8;  %individual weight of particles
    PHI2 = 0.8;  %social weight of particles
    W = 1.2;   %inertial factor

    vmin=-5;
    vmax=5;
    
    if size(I,3)==1 %grayscale image
        vR=zeros(N,N_PAR);  %velocities of particles
        X_MAXR = Lmax*ones(1,N_PAR);
        X_MINR = ones(1,N_PAR);
        gBestR = zeros(1,N_PAR);
        gbestvalueR = -10000;
        gauxR = ones(N,1);
        xBestR=zeros(N,N_PAR);
        fitBestR=zeros(N,1);
        fitR = zeros(N,1);
        xR = zeros(N,N_PAR);
        for i = 1: N
            for j = 1: N_PAR
                xR(i,j) = fix(rand(1,1) * ( X_MAXR(j)-X_MINR(j) ) + X_MINR(j));
            end
        end
        for si=1:length(xR)
           xR(si,:)=sort(xR(si,:)); 
        end
    elseif size(I,3)==3 %RGB image    
        vR=zeros(N,N_PAR);  %velocities of particles
        vG=zeros(N,N_PAR);
        vB=zeros(N,N_PAR);
        X_MAXR = Lmax*ones(1,N_PAR);
        X_MINR = ones(1,N_PAR);
        X_MAXG = Lmax*ones(1,N_PAR);
        X_MING = ones(1,N_PAR);
        X_MAXB = Lmax*ones(1,N_PAR);
        X_MINB = ones(1,N_PAR);
        gBestR = zeros(1,N_PAR);
        gbestvalueR = -10000;
        gauxR = ones(N,1);
        xBestR=zeros(N,N_PAR);
        fitBestR=zeros(N,1);
        fitR = zeros(N,1);
        gBestG = zeros(1,N_PAR);
        gbestvalueG = -10000;
        gauxG = ones(N,1);
        xBestG=zeros(N,N_PAR);
        fitBestG=zeros(N,1);
        fitG = zeros(N,1);
        gBestB = zeros(1,N_PAR);
        gbestvalueB = -10000;
        gauxB = ones(N,1);
        xBestB=zeros(N,N_PAR);
        fitBestB=zeros(N,1);
        fitB = zeros(N,1);
        xR = zeros(N,N_PAR);
        for i = 1: N
            for j = 1: N_PAR
                xR(i,j) = fix(rand(1,1) * ( X_MAXR(j)-X_MINR(j) ) + X_MINR(j));
            end
        end
        xG = zeros(N,N_PAR);
        for i = 1: N
            for j = 1: N_PAR
                xG(i,j) = fix(rand(1,1) * ( X_MAXG(j)-X_MING(j) ) + X_MING(j));
            end
        end
        xB = zeros(N,N_PAR);
        for i = 1: N
            for j = 1: N_PAR
                xB(i,j) = fix(rand(1,1) * ( X_MAXB(j)-X_MINB(j) ) + X_MINB(j));
            end
        end
        for si=1:length(xR)
           xR(si,:)=sort(xR(si,:)); 
        end
        for si=1:length(xG)
           xG(si,:)=sort(xG(si,:));
        end
        for si=1:length(xB)
           xB(si,:)=sort(xB(si,:)); 
        end
    end
    
    nger=1;

    for j=1:N
        if size(I,3)==1 %grayscale image
            fitR(j)=sum(probR(1:xR(j,1)))*(sum((1:xR(j,1)).*probR(1:xR(j,1))/sum(probR(1:xR(j,1)))) - sum((1:Lmax).*probR(1:Lmax)) )^2;
            for jlevel=2:level-1
                fitR(j)=fitR(j)+sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel)))*(sum((xR(j,jlevel-1)+1:xR(j,jlevel)).*probR(xR(j,jlevel-1)+1:xR(j,jlevel))/sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel))))- sum((1:Lmax).*probR(1:Lmax)))^2;
            end
            fitR(j)=fitR(j)+sum(probR(xR(j,level-1)+1:Lmax))*(sum((xR(j,level-1)+1:Lmax).*probR(xR(j,level-1)+1:Lmax)/sum(probR(xR(j,level-1)+1:Lmax)))- sum((1:Lmax).*probR(1:Lmax)))^2;
            fitBestR(j)=fitR(j);
        elseif size(I,3)==3 %RGB image
            fitR(j)=sum(probR(1:xR(j,1)))*(sum((1:xR(j,1)).*probR(1:xR(j,1))/sum(probR(1:xR(j,1)))) - sum((1:Lmax).*probR(1:Lmax)) )^2;
            for jlevel=2:level-1
                fitR(j)=fitR(j)+sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel)))*(sum((xR(j,jlevel-1)+1:xR(j,jlevel)).*probR(xR(j,jlevel-1)+1:xR(j,jlevel))/sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel))))- sum((1:Lmax).*probR(1:Lmax)))^2;
            end
            fitR(j)=fitR(j)+sum(probR(xR(j,level-1)+1:Lmax))*(sum((xR(j,level-1)+1:Lmax).*probR(xR(j,level-1)+1:Lmax)/sum(probR(xR(j,level-1)+1:Lmax)))- sum((1:Lmax).*probR(1:Lmax)))^2;
            fitBestR(j)=fitR(j);
            fitG(j)=sum(probG(1:xG(j,1)))*(sum((1:xG(j,1)).*probG(1:xG(j,1))/sum(probG(1:xG(j,1)))) - sum((1:Lmax).*probG(1:Lmax)) )^2;
            for jlevel=2:level-1
                fitG(j)=fitG(j)+sum(probG(xG(j,jlevel-1)+1:xG(j,jlevel)))*(sum((xG(j,jlevel-1)+1:xG(j,jlevel)).*probG(xG(j,jlevel-1)+1:xG(j,jlevel))/sum(probG(xG(j,jlevel-1)+1:xG(j,jlevel))))- sum((1:Lmax).*probG(1:Lmax)))^2;
            end
            fitG(j)=fitG(j)+sum(probG(xG(j,level-1)+1:Lmax))*(sum((xG(j,level-1)+1:Lmax).*probG(xG(j,level-1)+1:Lmax)/sum(probG(xG(j,level-1)+1:Lmax)))- sum((1:Lmax).*probG(1:Lmax)))^2;
            fitBestG(j)=fitG(j);
            fitB(j)=sum(probB(1:xB(j,1)))*(sum((1:xB(j,1)).*probB(1:xB(j,1))/sum(probB(1:xB(j,1)))) - sum((1:Lmax).*probB(1:Lmax)) )^2;
            for jlevel=2:level-1
                fitB(j)=fitB(j)+sum(probB(xB(j,jlevel-1)+1:xB(j,jlevel)))*(sum((xB(j,jlevel-1)+1:xB(j,jlevel)).*probB(xB(j,jlevel-1)+1:xB(j,jlevel))/sum(probB(xB(j,jlevel-1)+1:xB(j,jlevel))))- sum((1:Lmax).*probB(1:Lmax)))^2;
            end
            fitB(j)=fitB(j)+sum(probB(xB(j,level-1)+1:Lmax))*(sum((xB(j,level-1)+1:Lmax).*probB(xB(j,level-1)+1:Lmax)/sum(probB(xB(j,level-1)+1:Lmax)))- sum((1:Lmax).*probB(1:Lmax)))^2;
            fitBestB(j)=fitB(j);
        end
    end

    if size(I,3)==1 %grayscale image
        [aR,bR]=max(fitR);
        gBestR=xR(bR,:);
        gbestvalueR = fitR(bR);
        xBestR = xR;
    elseif size(I,3)==3 %RGB image
        [aR,bR]=max(fitR);
        gBestR=xR(bR,:);
        gbestvalueR = fitR(bR);
        [aG,bG]=max(fitG);
        gBestG=xG(bG,:);
        gbestvalueG = fitG(bG);
        [aB,bB]=max(fitB);
        gBestB=xR(bB,:);
        gbestvalueB = fitB(bB);
        xBestR = xR;
        xBestG = xG;
        xBestB = xB;
    end
    
    while(nger<=N_GER)
        i=1;
        
        randnum1 = rand ([N, N_PAR]);
        randnum2 = rand ([N, N_PAR]);
    
        if size(I,3)==1 %grayscale image
            vR = fix(W.*vR + randnum1.*(PHI1.*(xBestR-xR)) + randnum2.*(PHI2.*(gauxR*gBestR-xR)));
            vR = ( (vR <= vmin).*vmin ) + ( (vR > vmin).*vR );
            vR = ( (vR >= vmax).*vmax ) + ( (vR < vmax).*vR );
            xR = xR+vR;
        elseif size(I,3)==3 %RGB image    
            vR = fix(W.*vR + randnum1.*(PHI1.*(xBestR-xR)) + randnum2.*(PHI2.*(gauxR*gBestR-xR)));
            vR = ( (vR <= vmin).*vmin ) + ( (vR > vmin).*vR );
            vR = ( (vR >= vmax).*vmax ) + ( (vR < vmax).*vR );
            xR = xR+vR;
            vG = fix(W.*vG + randnum1.*(PHI1.*(xBestG-xG)) + randnum2.*(PHI2.*(gauxG*gBestG-xG)));
            vG = ( (vG <= vmin).*vmin ) + ( (vG > vmin).*vG );
            vG = ( (vG >= vmax).*vmax ) + ( (vG < vmax).*vG );
            xG = xG+vG;
            vB = fix(W.*vB + randnum1.*(PHI1.*(xBestB-xB)) + randnum2.*(PHI2.*(gauxB*gBestB-xB)));
            vB = ( (vB <= vmin).*vmin ) + ( (vB > vmin).*vB );
            vB = ( (vB >= vmax).*vmax ) + ( (vB < vmax).*vB );        
            xB = xB+vB;
        end
        
        if size(I,3)==1 %grayscale image
            xR = ( (xR <= X_MINR(1)).*X_MINR(1) ) + ( (xR > X_MINR(1)).*xR );
            xR = ( (xR >= X_MAXR(1)).*X_MAXR(1) ) + ( (xR < X_MAXR(1)).*xR );
        elseif size(I,3)==3 %RGB image  
            xR = ( (xR <= X_MINR(1)).*X_MINR(1) ) + ( (xR > X_MINR(1)).*xR );
            xR = ( (xR >= X_MAXR(1)).*X_MAXR(1) ) + ( (xR < X_MAXR(1)).*xR );
            xG = ( (xG <= X_MING(1)).*X_MING(1) ) + ( (xG > X_MING(1)).*xG );
            xG = ( (xG >= X_MAXG(1)).*X_MAXG(1) ) + ( (xG < X_MAXG(1)).*xG );
            xB = ( (xB <= X_MINB(1)).*X_MINB(1) ) + ( (xB > X_MINB(1)).*xB );
            xB = ( (xB >= X_MAXB(1)).*X_MAXB(1) ) + ( (xB < X_MAXB(1)).*xB );
        end
            

        for j = 1:N
            for k = 1:N_PAR
                if size(I,3)==1 %grayscale image
                    if (k==1)&&(k~=N_PAR)
                        if xR(j,k) < X_MINR(k)
                            xR(j,k) = X_MINR(k);
                        elseif xR(j,k) > xR(j,k+1)
                            xR(j,k) = xR(j,k+1);
                        end
                    end
                    if ((k>1)&&(k<N_PAR))
                        if xR(j,k) < xR(j,k-1)
                            xR(j,k) = xR(j,k-1);
                        elseif xR(j,k) > xR(j,k+1)
                            xR(j,k) = xR(j,k+1);
                        end
                    end
                    if (k==N_PAR)&&(k~=1)
                        if xR(j,k) < xR(j,k-1)
                            xR(j,k) = xR(j,k-1);
                        elseif xR(j,k) > X_MAXR(k)
                            xR(j,k) = X_MAXR(k);
                        end
                    end
                    if (k==1)&&(k==N_PAR)
                        if xR(j,k) < X_MINR(k)
                            xR(j,k) = X_MINR(k);
                        elseif xR(j,k) > X_MAXR(k)
                            xR(j,k) = X_MAXR(k);
                        end
                    end
                elseif size(I,3)==3 %RGB image     
                    if (k==1)&&(k~=N_PAR)
                        if xR(j,k) < X_MINR(k)
                            xR(j,k) = X_MINR(k);
                        elseif xR(j,k) > xR(j,k+1)
                            xR(j,k) = xR(j,k+1);
                        end
                    end
                    if ((k>1)&&(k<N_PAR))
                        if xR(j,k) < xR(j,k-1)
                            xR(j,k) = xR(j,k-1);
                        elseif xR(j,k) > xR(j,k+1)
                            xR(j,k) = xR(j,k+1);
                        end
                    end
                    if (k==N_PAR)&&(k~=1)
                        if xR(j,k) < xR(j,k-1)
                            xR(j,k) = xR(j,k-1);
                        elseif xR(j,k) > X_MAXR(k)
                            xR(j,k) = X_MAXR(k);
                        end
                    end
                    if (k==1)&&(k==N_PAR)
                        if xR(j,k) < X_MINR(k)
                            xR(j,k) = X_MINR(k);
                        elseif xR(j,k) > X_MAXR(k)
                            xR(j,k) = X_MAXR(k);
                        end
                    end
                    if (k==1)&&(k~=N_PAR)
                        if xG(j,k) < X_MING(k)
                            xG(j,k) = X_MING(k);
                        elseif xG(j,k) > xG(j,k+1)
                            xG(j,k) = xG(j,k+1);
                            %                         disp ('passou o max');
                        end
                    end
                    if ((k>1)&&(k<N_PAR))
                        if xG(j,k) < xG(j,k-1)
                            xG(j,k) = xG(j,k-1);
                            %                         disp ('passou o min');
                        elseif xG(j,k) > xG(j,k+1)
                            xG(j,k) = xG(j,k+1);
                            %                         disp ('passou o max');
                        end
                    end
                    if (k==N_PAR)&&(k~=1)
                        if xG(j,k) < xG(j,k-1)
                            xG(j,k) = xG(j,k-1);
                            %                         disp ('passou o min');
                        elseif xG(j,k) > X_MAXG(k)
                            xG(j,k) = X_MAXG(k);
                            %                         disp ('passou o max');
                        end
                    end
                    if (k==1)&&(k==N_PAR)
                        if xG(j,k) < X_MING(k)
                            xG(j,k) = X_MING(k);
                        elseif xG(j,k) > X_MAXG(k)
                            xG(j,k) = X_MAXG(k);
                        end
                    end
                    if (k==1)&&(k~=N_PAR)
                        if xB(j,k) < X_MINB(k)
                            xB(j,k) = X_MINB(k);
                            %                         disp ('passou o min');
                        elseif xB(j,k) > xB(j,k+1)
                            xB(j,k) = xB(j,k+1);
                            %                         disp ('passou o max');
                        end
                    end
                    if ((k>1)&&(k<N_PAR))
                        if xB(j,k) < xB(j,k-1)
                            xB(j,k) = xB(j,k-1);
                            %                         disp ('passou o min');
                        elseif xB(j,k) > xB(j,k+1)
                            xB(j,k) = xB(j,k+1);
                            %                         disp ('passou o max');
                        end
                    end
                    if (k==N_PAR)&&(k~=1)
                        if xB(j,k) < xB(j,k-1)
                            xB(j,k) = xB(j,k-1);
                            %                         disp ('passou o min');
                        elseif xB(j,k) > X_MAXB(k)
                            xB(j,k) = X_MAXB(k);
                            %                         disp ('passou o max');
                        end
                    end
                    if (k==1)&&(k==N_PAR)
                        if xB(j,k) < X_MINB(k)
                            xB(j,k) = X_MINB(k);
                        elseif xB(j,k) > X_MAXB(k)
                            xB(j,k) = X_MAXB(k);
                        end
                    end
                end
            end
        end

        while(i<=N)
            if(i==N)
                for j=1:N
                    if size(I,3)==1 %grayscale image
                        fitR(j)=sum(probR(1:xR(j,1)))*(sum((1:xR(j,1)).*probR(1:xR(j,1))/sum(probR(1:xR(j,1)))) - sum((1:Lmax).*probR(1:Lmax)) )^2;
                        for jlevel=2:level-1
                            fitR(j)=fitR(j)+sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel)))*(sum((xR(j,jlevel-1)+1:xR(j,jlevel)).*probR(xR(j,jlevel-1)+1:xR(j,jlevel))/sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel))))- sum((1:Lmax).*probR(1:Lmax)))^2;
                        end
                        fitR(j)=fitR(j)+sum(probR(xR(j,level-1)+1:Lmax))*(sum((xR(j,level-1)+1:Lmax).*probR(xR(j,level-1)+1:Lmax)/sum(probR(xR(j,level-1)+1:Lmax)))- sum((1:Lmax).*probR(1:Lmax)))^2;
                        if fitR(j) > fitBestR(j)
                            fitBestR(j) = fitR(j);
                            xBestR(j,:) = xR(j,:);
                        end
                    elseif size(I,3)==3 %RGB image
                        fitR(j)=sum(probR(1:xR(j,1)))*(sum((1:xR(j,1)).*probR(1:xR(j,1))/sum(probR(1:xR(j,1)))) - sum((1:Lmax).*probR(1:Lmax)) )^2;
                        for jlevel=2:level-1
                            
                            fitR(j)=fitR(j)+sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel)))*(sum((xR(j,jlevel-1)+1:xR(j,jlevel)).*probR(xR(j,jlevel-1)+1:xR(j,jlevel))/sum(probR(xR(j,jlevel-1)+1:xR(j,jlevel))))- sum((1:Lmax).*probR(1:Lmax)))^2;
                        end
                        fitR(j)=fitR(j)+sum(probR(xR(j,level-1)+1:Lmax))*(sum((xR(j,level-1)+1:Lmax).*probR(xR(j,level-1)+1:Lmax)/sum(probR(xR(j,level-1)+1:Lmax)))- sum((1:Lmax).*probR(1:Lmax)))^2;
                        fitG(j)=sum(probG(1:xG(j,1)))*(sum((1:xG(j,1)).*probG(1:xG(j,1))/sum(probG(1:xG(j,1)))) - sum((1:Lmax).*probG(1:Lmax)) )^2;
                        for jlevel=2:level-1
                            fitG(j)=fitG(j)+sum(probG(xG(j,jlevel-1)+1:xG(j,jlevel)))*(sum((xG(j,jlevel-1)+1:xG(j,jlevel)).*probG(xG(j,jlevel-1)+1:xG(j,jlevel))/sum(probG(xG(j,jlevel-1)+1:xG(j,jlevel))))- sum((1:Lmax).*probG(1:Lmax)))^2;
                        end
                        fitG(j)=fitG(j)+sum(probG(xG(j,level-1)+1:Lmax))*(sum((xG(j,level-1)+1:Lmax).*probG(xG(j,level-1)+1:Lmax)/sum(probG(xG(j,level-1)+1:Lmax)))- sum((1:Lmax).*probG(1:Lmax)))^2;
                        fitB(j)=sum(probB(1:xB(j,1)))*(sum((1:xB(j,1)).*probB(1:xB(j,1))/sum(probB(1:xB(j,1)))) - sum((1:Lmax).*probB(1:Lmax)) )^2;
                        for jlevel=2:level-1
                            
                            fitB(j)=fitB(j)+sum(probB(xB(j,jlevel-1)+1:xB(j,jlevel)))*(sum((xB(j,jlevel-1)+1:xB(j,jlevel)).*probB(xB(j,jlevel-1)+1:xB(j,jlevel))/sum(probB(xB(j,jlevel-1)+1:xB(j,jlevel))))- sum((1:Lmax).*probB(1:Lmax)))^2;
                        end
                        fitB(j)=fitB(j)+sum(probB(xB(j,level-1)+1:Lmax))*(sum((xB(j,level-1)+1:Lmax).*probB(xB(j,level-1)+1:Lmax)/sum(probB(xB(j,level-1)+1:Lmax)))- sum((1:Lmax).*probB(1:Lmax)))^2;
                        if fitR(j) > fitBestR(j)
                            fitBestR(j) = fitR(j);
                            xBestR(j,:) = xR(j,:);
                        end
                        if fitG(j) > fitBestG(j)
                            fitBestG(j) = fitG(j);
                            xBestG(j,:) = xG(j,:);
                        end
                        if fitB(j) > fitBestB(j)
                            fitBestB(j) = fitB(j);
                            xBestB(j,:) = xB(j,:);
                        end
                    end
                end
                
                if size(I,3)==1 %grayscale image
                    [aR,bR] = max (fitR);
                    if (fitR(bR) > gbestvalueR)
                        gBestR=xR(bR,:)-1;
                        gbestvalueR = fitR(bR);
                    end
                elseif size(I,3)==3 %RGB image
                    [aR,bR] = max (fitR);
                    [aG,bG] = max (fitG);
                    [aB,bB] = max (fitB);
                    if (fitR(bR) > gbestvalueR)
                        gBestR=xR(bR,:)-1;
                        gbestvalueR = fitR(bR);
                    end
                    if (fitG(bG) > gbestvalueG)
                        gBestG=xG(bG,:)-1;
                        gbestvalueG = fitG(bG);
                    end
                    if (fitB(bB) > gbestvalueB)
                        gBestB=xB(bB,:)-1;
                        gbestvalueB = fitB(bB);
                    end
                end
                nger=nger+1;
            end
            i=i+1;
        end
    end
%     gbestvalueR
Iout1=edge(I);
end


if size(I,3)==1 %grayscale image
    gBestR=sort(gBestR);
    Iout=imageGRAY(I,gBestR);
elseif size(I,3)==3 %RGB image
    gBestR=sort(gBestR);
    gBestG=sort(gBestG);
    gBestB=sort(gBestB);
    Iout=imageRGB(I,gBestR,gBestG,gBestB);
end

if nargout>1
    if size(I,3)==1 %grayscale image
        gBestR=sort(gBestR);
        intensity=gBestR;     %return optimal intensity
        if nargout>2
            fitness=gbestvalueR;    %return fitness value
            if nargout>3
                time=toc;   %return CPU time
            end
        end
    elseif size(I,3)==3 %RGB image
        gBestR=sort(gBestR);
        gBestG=sort(gBestG);
        gBestB=sort(gBestB);
        intensity=[gBestR; gBestG; gBestB];
        if nargout>2
            fitness=[gbestvalueR; gbestvalueG; gbestvalueB];    %return fitness value
            if nargout>3
                time=toc;   %return CPU time
            end
        end
    end
end


function imgOut=imageRGB(img,Rvec,Gvec,Bvec)
imgOutR=img(:,:,1);
imgOutG=img(:,:,2);
imgOutB=img(:,:,3);

Rvec=[0 Rvec 256];
for iii=1:size(Rvec,2)-1
    at=find(imgOutR(:,:)>=Rvec(iii) & imgOutR(:,:)<Rvec(iii+1));
    imgOutR(at)=Rvec(iii);
end

Gvec=[0 Gvec 256];
for iii=1:size(Gvec,2)-1
    at=find(imgOutG(:,:)>=Gvec(iii) & imgOutG(:,:)<Gvec(iii+1));
    imgOutG(at)=Gvec(iii);
end

Bvec=[0 Bvec 256];
for iii=1:size(Bvec,2)-1
    at=find(imgOutB(:,:)>=Bvec(iii) & imgOutB(:,:)<Bvec(iii+1));
    imgOutB(at)=Bvec(iii);
end

imgOut=img;

imgOut(:,:,1)=imgOutR;
imgOut(:,:,2)=imgOutG;
imgOut(:,:,3)=imgOutB;


function imgOut=imageGRAY(img,Rvec)
% imgOut=img;
limites=[0 Rvec 255];
tamanho=size(img);
imgOut(:,:)=img*0;
% cores=[ 0   0   0;
%         255 0   0;
%         0   255 0;
%         0   0   255;
%         255 255 0;
%         0   255 255;
%         255 0   255;
%         255 255 255];
        
cores=colormap(lines)*255;
k=1;
    for i= 1:tamanho(1,1)
        for j=1:tamanho(1,2)
            while(k<size(limites,2))
                if(img(i,j)>=limites(1,k) && img(i,j)<=limites(1,k+1))
                    imgOut(i,j,1)=limites(1,k);
%                     imgOut(i,j,2)=cores(k,2);
%                     imgOut(i,j,3)=cores(k,3);
                end
                k=k+1;
            end
            k=1;
        end
    end

    
function [C,RA,RB] = insertrows(A,B,ind)
error(nargchk(2,3,nargin)) ;

if nargin==2,
    % just horizontal concatenation, suggested by Tim Davis
    ind = size(A,1) ;
end

% shortcut when any of the inputs are empty
if isempty(B) || isempty(ind),    
    C = A ;     
    if nargout > 1,
        RA = 1:size(A,1) ;
        RB = [] ;
    end
    return
end

sa = size(A) ;

% match the sizes of A, B
if numel(B)==1,
    % B has a single argument, expand to match A
    sb = [1 sa(2:end)] ;
    B = repmat(B,sb) ;
else
    % otherwise check for dimension errors
    if ndims(A) ~= ndims(B),
        error('insertrows:DimensionMismatch', ...
            'Both input matrices should have the same number of dimensions.') ;
    end
    sb = size(B) ;
    if ~all(sa(2:end) == sb(2:end)),
        error('insertrows:DimensionMismatch', ...
            'Both input matrices should have the same number of columns (and planes, etc).') ;
    end
end

ind = ind(:) ; % make as row vector
ni = length(ind) ;

% match the sizes of B and IND
if ni ~= sb(1),
    if ni==1 && sb(1) > 1,
        % expand IND
        ind = repmat(ind,sb(1),1) ;
    elseif (ni > 1) && (sb(1)==1),
        % expand B
        B = repmat(B,ni,1) ;
    else
        error('insertrows:InputMismatch',...
            'The number of rows to insert should equal the number of insertion positions.') ;
    end
end

sb = size(B) ;

% the actual work
% 1. concatenate matrices
C = [A ; B] ;
% 2. sort the respective indices, the first output of sort is ignored (by
% giving it the same name as the second output, one avoids an extra 
% large variable in memory)
[abi,abi] = sort([[1:sa(1)].' ; ind(:)]) ;
% 3. reshuffle the large matrix
C = C(abi,:) ;
% 4. reshape as A for nd matrices (nd>2)
if ndims(A) > 2,
    sc = sa ;
    sc(1) = sc(1)+sb(1) ;
    C = reshape(C,sc) ;
end

if nargout > 1,
    % additional outputs required
    R = [zeros(sa(1),1) ; ones(sb(1),1)] ;
    R = R(abi) ;
    RA = find(R==0) ;
    RB = find(R==1) ;
end


    
    
    