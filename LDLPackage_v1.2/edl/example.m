addpath('../');
addpath('../measures');

clear all;
close all;
clc;

is_doc2vec = true;

root_path = '../../data/AffectiveText.Semeval.2007';

test_result = CVResult;

file_path = fullfile(root_path, '');

% cnn
[precision, recall, accuracy, F1, dist_name, distance] = ComputeEDLMetrics(fullfile(file_path, 'result.mat'));
test_result = CVResultAppend(test_result, precision, recall, accuracy, F1, distance, dist_name);   

fprintf('*******************************************************\n\n\n');
fprintf('cnn\n');
CVResultDisplay(test_result);

