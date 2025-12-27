% export_stimuli_v13_SEED_FIX.m
% Fixes the Random Number Generator initialization bug.
% Guarantees synchronization by clearing persistent memory and forcing negative seed.

% --- 1. CONFIGURATION ---
base_path = '/Users/jonathandadcha/Desktop/Retina-Comp-Project/data/10.12751_g-node.2j3d2i'; 
session_name = '20171116_sr_le_fp'; 

% --- 2. SETUP PATHS ---
code_folder = fullfile(base_path, 'code_for_stim_reconstruction');
addpath(genpath(code_folder));

full_session_path = fullfile(base_path, session_name);
output_dir = fullfile(full_session_path, 'processed_data');
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

stim_dir = fullfile(full_session_path, 'stimuli');
ft_dir = fullfile(full_session_path, 'frametimes');

disp(['Working on session: ' session_name]);

% --- 3. NATURAL IMAGES ---
disp(' ');
disp('--- Processing Natural Images ---');
try
    stim_files = dir(fullfile(stim_dir, '*imgseq*.mat'));
    ft_files = dir(fullfile(ft_dir, '*imgseq*.mat'));
    if isempty(ft_files), ft_files = dir(fullfile(ft_dir, '*imseq*.mat')); end
    
    if ~isempty(stim_files) && ~isempty(ft_files)
        stim_target = stim_files(1).name;
        ft_target = ft_files(1).name;
        load(fullfile(stim_dir, stim_target), 'stimpara');
        load(fullfile(ft_dir, ft_target), 'ftimes');
        
        % FIX: Clear persistent memory before starting
        clear ran1; 
        % FIX: Force negative seed for initialization
        ran1(-abs(stimpara.seed)); 
        
        prs = [];
        Nimages = 300;
        for r = 1:stimpara.nrepeats
            temp_seq = 1:Nimages;
            for i = length(temp_seq):-1:2
                ridx = floor(ran1(0) * i) + 1;
                temp = temp_seq(i); temp_seq(i) = temp_seq(ridx); temp_seq(ridx) = temp;
            end
            prs = [prs, 0, temp_seq];
        end
        if length(prs) > length(ftimes), prs = prs(1:length(ftimes)); end
        
        h5_filename = fullfile(output_dir, 'natural_scenes_metadata.h5');
        if exist(h5_filename, 'file'), delete(h5_filename); end
        hdf5write(h5_filename, '/image_order', prs);
        hdf5write(h5_filename, '/frame_times', ftimes, 'WriteMode', 'append');
        disp(['SUCCESS: Metadata saved.']);
    end
catch ME
    disp(['Error in Natural Images: ' ME.message]);
end

% --- 4. WHITE NOISE (THE FIX) ---
disp(' ');
disp('--- Processing White Noise (SEED FIXED) ---');
try
    wn_files = dir(fullfile(stim_dir, '*checkerflicker*.mat'));
    if ~isempty(wn_files)
        load(fullfile(stim_dir, wn_files(1).name), 'stimpara');
        
        Nx = stimpara.Nx;
        Ny = stimpara.Ny;
        Nframes_full = 75000;
        seed = stimpara.seed;
        
        disp(['Generating ' num2str(Nframes_full) ' frames...']);
        
        % --- CRITICAL FIXES START HERE ---
        % 1. Clear previous state of ran1 from memory
        clear ran1;
        
        % 2. Force seed to be negative to trigger initialization in ran1.m
        % Even if stimpara.seed is positive, ran1 needs a negative input to reset.
        actual_seed = -abs(seed);
        ran1(actual_seed);
        
        disp(['RNG Initialized with seed: ' num2str(actual_seed)]);
        % ---------------------------------
        
        movie_chunk = zeros(Ny, Nx, Nframes_full); 
        
        % C++ Order: Time -> Y -> X
        for t = 1:Nframes_full
            for y = 1:Ny
                for x = 1:Nx
                    r = ran1(0);
                    if r > 0.5
                        movie_chunk(y, x, t) = 1;
                    else
                        movie_chunk(y, x, t) = 0;
                    end
                end
            end
            if mod(t, 10000) == 0, disp(['... ' num2str(t)]); end
        end
        
        wn_h5_filename = fullfile(output_dir, 'white_noise_full.h5');
        if exist(wn_h5_filename, 'file'), delete(wn_h5_filename); end
        hdf5write(wn_h5_filename, '/stimulus', movie_chunk);
        
        disp(['SUCCESS: Full video saved to ' wn_h5_filename]);
    else
        warning('No checkerflicker file found.');
    end
catch ME
    disp(['Error in White Noise: ' ME.message]);
end
disp('--- DONE ---');