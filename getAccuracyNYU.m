function getAccuracyNYU

voidClass=0
numClasses = 13;

globalacc = 0;
totalpoints = 0;
voidpoints = 0;

num_imgs = 653;


cf = zeros(numClasses);

for i = 0 : num_imgs
	
    i
    
    NYU_TEST_ANNOTATION = sprintf('/scratch/NYU_TEST/labels/Ordered/new_nyu_test_class13_%04d.png',i);

    Ia = imread(NYU_TEST_ANNOTATION);
    
    Ia = imresize(Ia,[224 224],'nearest');
    
    %Ia = Ia .* sum(segImg,3);%to ignore all zero annotations.
    Ivoid = Ia == voidClass;% NYU
    voidpoints = voidpoints + sum(sum(Ivoid));

    Ia = Ia .* uint8(not(Ivoid));%ignore computing accuracy at void indices
    totalpoints = totalpoints + sum(Ia(:)>0);

    %Use your predictions here
    NYU_TEST_PREDICTION = sprintf('/scratch/DeconvNetFloat/evaluation/new_nyu_class13_eigen_acc/new_nyu_class13_eigen_%04d.png',i);

    Ip = imread(NYU_TEST_PREDICTION);
    if ( size(Ip,3) > 1 ) 
        Ip = rgb2gray(Ip);
    end
    
    %Ip = Ip .* sum(annot,3);%to ignore all zero annotations.
    Ip(Ivoid) = voidClass;
    % CamVid type accuracy
    for j = 1:numClasses
        for k = 1:numClasses
           if k ~= voidClass
               c1  = Ia == j;
               c1p = Ip == k;
               index = c1 .* c1p;
               cf(j,k) = cf(j,k) + sum(index(:));
           end
        end
        if j ~= voidClass
            c1  = Ia == j;
            c1p = Ip == j;
            index = c1 .* c1p;
            globalacc = globalacc + sum(index(:));
        end
    end

end

% confusion matrix
conf = zeros(numClasses);
for i = 1:numClasses
    if i ~= voidClass && sum(cf(i,:)) > 0
        conf(i,:) = cf(i,:)/sum(cf(i,:));
    end
end


conf

diag(conf)

globalacc = globalacc/totalpoints
classavg  = sum(diag(conf))/(numClasses)

