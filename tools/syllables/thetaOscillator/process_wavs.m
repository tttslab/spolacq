% Perform syllable segmentation for all the files in wavs.list.
% Herman Kamper, kamperh@gmail.com, 2015.

% clear all;

% Read list of wav files
fid = fopen('wavlist.list');
wav_files = textscan(fid, '%s', 'Delimiter', '\n');
wav_files = wav_files{1};
% wav_files ='combined_sounds.wav';
fclose(fid);

% 2) Generate Gammatone filterbank center frequencies (log-spacing)
minfreq = 50;
maxfreq = 7500;
bands = 20;

cfs = zeros(bands,1);
const = (maxfreq/minfreq)^(1/(bands-1));

cfs(1) = 50;
for k = 1:bands-1
    cfs(k+1) = cfs(k).*const;
end


envelopes = cell(size(wav_files));

for i = 1:length(wav_files)
    disp([int2str(i) ': ' wav_files{i}]);

    % 1) Load audio file    
    [x,fs] = audioread(wav_files{i});

    if(fs ~= 16000)
        x = resample(x,16000,fs);
        fs = 16000;
    end

    % 3) Compute gammatone envelopes and downsample to 1000 Hz

    env = zeros(length(x),length(cfs));
    for cf = 1:length(cfs)
        [~, env(:,cf), ~, ~] = gammatone_c(x, fs,cfs(cf));
    end
    envelopes{i} = resample(env,1000,fs);
end

% 4) Run oscillator-based segmentation
Q_value = 0.5;  % Q-value of the oscillator, default = 0.5 = critical damping
center_frequency = 5; % in Hz
threshold = 0.01;

[bounds,bounds_t,osc_env,nucleii] = thetaOscillator(envelopes,center_frequency,Q_value,threshold);

save('results/bounds_t.mat', 'wav_files', 'bounds_t')

fid = fopen('results/output.csv','w');
for i = 1:length(bounds_t)
    for k = 1:length(bounds_t{i})
        fprintf(fid,'%f ',bounds_t{i}(k));
    end
    fprintf(fid,'\n');
end
fclose(fid);

return;

% 6) Play syllables
for k = 1:length(bounds_t{1})-1
    disp('press button to hear the next syllable');
    pause;
    soundsc(x(bounds_t{1}(k)*fs:min(length(x),bounds_t{1}(k+1)*fs)),fs);
end
%}
