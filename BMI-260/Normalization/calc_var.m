% Clear all previous data
clc, clear all, close all;


%% Display results of each method
verbose = 0;


%% Load Source & Target images



mean_R_og = 0;
mean_G_og = 0;
mean_B_og = 0;

var_R_og = 0;
var_G_og = 0;
var_B_og = 0;

mean_R_HS = 0;
mean_G_HS = 0;
mean_B_HS = 0;

var_R_HS = 0;
var_G_HS = 0;
var_B_HS = 0;

mean_R_MM = 0;
mean_G_MM = 0;
mean_B_MM = 0;

var_R_MM = 0;
var_G_MM = 0;
var_B_MM = 0;

mean_R_RH = 0;
mean_G_RH = 0;
mean_B_RH = 0;

var_R_RH = 0;
var_G_RH = 0;
var_B_RH = 0;

mean_R_SM = 0;
mean_G_SM = 0;
mean_B_SM = 0;

var_R_SM = 0;
var_G_SM = 0;
var_B_SM = 0;

sdirectory = 'nuclei';
tifffiles = dir([sdirectory '/*.tif']);

for k = 1:length(tifffiles)
    disp('Top of loop');
    disp(k);
    disp(tifffiles(k).name);
    disp(length(tifffiles));
    
    filename = [sdirectory '/' tifffiles(k).name];
    SourceImage = imread(filename);
    
    RSource = SourceImage(:, :, 1);
    GSource = SourceImage(:, :, 2);
    BSource = SourceImage(:, :, 3);
    
    mean_R_og= mean_R_og + mean(RSource(:));
    mean_G_og= mean_G_og + mean(GSource(:)); 
    mean_B_og= mean_B_og + mean(BSource(:));
    
    var_R_og = var_R_og+var(double(RSource(:))); 
    var_G_og = var_G_og+var(double(GSource(:))); 
    var_B_og = var_B_og+var(double(BSource(:)));
    
end

var_R_og = var_R_og/length(tifffiles);
var_G_og = var_G_og/length(tifffiles); 
var_B_og = var_B_og/length(tifffiles);

% std_val = sqrt(var)
mean_R_og = mean_R_og/length(tifffiles);
mean_G_og = mean_G_og/length(tifffiles);
mean_B_og = mean_B_og/length(tifffiles);


%% HS

sdirectory = 'nuclei/output/NormHS';
tifffiles = dir([sdirectory '/*.tif']);

for k = 1:length(tifffiles)
    disp('Top of loop');
    disp(k);
    disp(tifffiles(k).name);
    disp(length(tifffiles));
    
    filename = [sdirectory '/' tifffiles(k).name];
    SourceImage = imread(filename);
    
    RSource = SourceImage(:, :, 1);
    GSource = SourceImage(:, :, 2);
    BSource = SourceImage(:, :, 3);
    
    mean_R_HS= mean_R_HS + mean(RSource(:));
    mean_G_HS= mean_G_HS + mean(GSource(:)); 
    mean_B_HS= mean_B_HS + mean(BSource(:));
    
    var_R_HS = var_R_HS+var(double(RSource(:))); 
    var_G_HS = var_G_HS+var(double(GSource(:))); 
    var_B_HS = var_B_HS+var(double(BSource(:)));
    
end

var_R_HS = var_R_HS/length(tifffiles);
var_G_HS = var_G_HS/length(tifffiles); 
var_B_HS = var_B_HS/length(tifffiles);

% std_val = sqrt(var)
mean_R_HS = mean_R_HS/length(tifffiles);
mean_G_HS = mean_G_HS/length(tifffiles);
mean_B_HS = mean_B_HS/length(tifffiles);

%% MM

sdirectory = 'nuclei/output/NormMM';
tifffiles = dir([sdirectory '/*.tif']);

for k = 1:length(tifffiles)
    disp('Top of loop');
    disp(k);
    disp(tifffiles(k).name);
    disp(length(tifffiles));
    
    filename = [sdirectory '/' tifffiles(k).name];
    SourceImage = imread(filename);
    
    RSource = SourceImage(:, :, 1);
    GSource = SourceImage(:, :, 2);
    BSource = SourceImage(:, :, 3);
    
    mean_R_MM= mean_R_MM + mean(RSource(:));
    mean_G_MM= mean_G_MM + mean(GSource(:)); 
    mean_B_MM= mean_B_MM + mean(BSource(:));
    
    var_R_MM = var_R_MM+var(double(RSource(:))); 
    var_G_MM = var_G_MM+var(double(GSource(:))); 
    var_B_MM = var_B_MM+var(double(BSource(:)));
    
end

var_R_MM = var_R_MM/length(tifffiles);
var_G_MM = var_G_MM/length(tifffiles); 
var_B_MM = var_B_MM/length(tifffiles);

% std_val = sqrt(var)
mean_R_MM = mean_R_MM/length(tifffiles);
mean_G_MM = mean_G_MM/length(tifffiles);
mean_B_MM = mean_B_MM/length(tifffiles);

%% RH

sdirectory = 'nuclei/output/NormRH';
tifffiles = dir([sdirectory '/*.tif']);

for k = 1:length(tifffiles)
    disp('Top of loop');
    disp(k);
    disp(tifffiles(k).name);
    disp(length(tifffiles));
    
    filename = [sdirectory '/' tifffiles(k).name];
    SourceImage = imread(filename);
    
    RSource = SourceImage(:, :, 1);
    GSource = SourceImage(:, :, 2);
    BSource = SourceImage(:, :, 3);
    
    mean_R_RH= mean_R_RH + mean(RSource(:));
    mean_G_RH= mean_G_RH + mean(GSource(:)); 
    mean_B_RH= mean_B_RH + mean(BSource(:));
    
    var_R_RH = var_R_RH+var(double(RSource(:))); 
    var_G_RH = var_G_RH+var(double(GSource(:))); 
    var_B_RH = var_B_RH+var(double(BSource(:)));
    
end

var_R_RH = var_R_RH/length(tifffiles);
var_G_RH = var_G_RH/length(tifffiles); 
var_B_RH = var_B_RH/length(tifffiles);

% std_val = sqrt(var)
mean_R_RH = mean_R_RH/length(tifffiles);
mean_G_RH = mean_G_RH/length(tifffiles);
mean_B_RH = mean_B_RH/length(tifffiles);

%% SM

sdirectory = 'nuclei/output/NormSM';
tifffiles = dir([sdirectory '/*.tif']);

for k = 1:length(tifffiles)
    disp('Top of loop');
    disp(k);
    disp(tifffiles(k).name);
    disp(length(tifffiles));
    
    filename = [sdirectory '/' tifffiles(k).name];
    SourceImage = imread(filename);
    
    RSource = SourceImage(:, :, 1);
    GSource = SourceImage(:, :, 2);
    BSource = SourceImage(:, :, 3);
    
    mean_R_SM= mean_R_SM + mean(RSource(:));
    mean_G_SM= mean_G_SM + mean(GSource(:)); 
    mean_B_SM= mean_B_SM + mean(BSource(:));
    
    var_R_SM = var_R_SM+var(double(RSource(:))); 
    var_G_SM = var_G_SM+var(double(GSource(:))); 
    var_B_SM = var_B_SM+var(double(BSource(:)));
    
end

var_R_SM = var_R_SM/length(tifffiles);
var_G_SM = var_G_SM/length(tifffiles); 
var_B_SM = var_B_SM/length(tifffiles);

% std_val = sqrt(var)
mean_R_SM = mean_R_SM/length(tifffiles);
mean_G_SM = mean_G_SM/length(tifffiles);
mean_B_SM = mean_B_SM/length(tifffiles);


comp_mean = [mean_R_og mean_R_HS mean_R_RH mean_R_MM mean_R_SM;
            mean_G_og mean_G_HS mean_G_RH mean_G_MM mean_G_SM;
            mean_B_og mean_B_HS mean_B_RH mean_B_MM mean_B_SM]

comp_var = [var_R_og var_R_HS var_R_RH var_R_MM var_R_SM;
            var_G_og var_G_HS var_G_RH var_G_MM var_G_SM;
            var_B_og var_B_HS var_B_RH var_B_MM var_B_SM];   
        
comp_std = sqrt(comp_var)

figure;
hold on
bar(1:3,comp_mean)

for i = 1:3
    j = 1:5; 
    x = -0.5 + i + 1/6 * j; 
    errorbar(x,comp_mean(i,j),comp_std(i,j),'.')
end



