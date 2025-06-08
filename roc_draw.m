function ROC_data = roc_draw(data, dispp, dispt)

    L = size(data,1);                 
    s_data = unique(sort(data(:)));     
    d_data = diff(s_data);              
    if(isempty(d_data)), error('Both class data are the same!'); end
    d_data(length(d_data)+1,1) = d_data(length(d_data));
    satur(1,1) = s_data(1) - d_data(1);               
    satur(2:length(s_data)+1,1) = s_data + d_data./2; 
    
    
    if(mean(data(:,1))>mean(data(:,2))), data = [data(:,2),data(:,1)]; end
        
    fitting = zeros(size(satur,1),2);
    distance = zeros(size(satur,1),1);
    for id_t = 1:1:length(satur)
        true_positive = length(find(data(:,2) >= satur(id_t)));    
        false_positive = length(find(data(:,1) >= satur(id_t)));    
        false_negative = L - true_positive;                                    
        true_negative = L - false_positive;                                    
        
        fitting(id_t,1) = true_positive/(true_positive + false_negative);   
        fitting(id_t,2) = true_negative/(true_negative + false_positive);	
        
       
        distance(id_t)= sqrt((1-fitting(id_t,1))^2+(fitting(id_t,2)-1)^2);
    end
    
    
    [~, opt] = min(distance);
    true_positive = length(find(data(:,2) >= satur(opt)));   
    false_positive = length(find(data(:,1) >= satur(opt)));    
    false_negative = L - true_positive;                                    
    true_negative = L - false_positive;                                    
        
    param.Threshold = satur(opt);       
    param.Sensi = fitting(opt,1);         
    param.Speci = fitting(opt,2);         
    param.absoulte  = abs(trapz(1-fitting(:,2), fitting(:,1))); 
    param.Accuracy = (true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative);             
    param.posit_poi_val   = true_positive/(true_positive+false_positive);           
    param.negati_poi_val   = true_negative/(true_negative+false_negative);           

   
    if(dispp == 1)
          fill([1-fitting(:,2); 1], [fitting(:,1); 0],'m');
        hold on; plot(1-fitting(:,2), fitting(:,1), '-r', 'LineWidth', 2);
        hold on; plot(1-fitting(opt,2), fitting(opt,1), '-+k', 'MarkerSize', 10);
        hold on; plot(1-fitting(opt,2), fitting(opt,1), '-xk', 'MarkerSize', 12);
        hold off; 
        axis square;
        grid on;
        xlabel('1 - specificity');
        ylabel('sensibility');
        title(['Absolute ROC = ' num2str(param.absoulte)]);
    end
       
    
    ROC_data.param = param;
    ROC_data.fitting = fitting;
end