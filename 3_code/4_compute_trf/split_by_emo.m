close all
clear all

% =========================
% Subjects
% =========================
subjects = {'sub-pilot_Jun'};

% =========================
% Paths
% =========================
base_indir = "C:\projects\emo_EEG\data\processed";
outdir = "C:\projects\emo_EEG\data_pipeline\visualization_analysis\prp\CI";

if ~exist(outdir, "dir")
    mkdir(outdir);
end

% =========================
% Settings
% =========================
emotions_to_save = ["hap", "sad"];

% Choose: "ica", "ica2", or "both"
ica_mode = "ica";

switch ica_mode
    case "ica"
        file_patterns = ["*_ica_prp_detrendLinear.mat"];
    case "ica2"
        file_patterns = ["*_ica2_prp_detrendLinear.mat"];
    case "both"
        file_patterns = ["*_ica_prp_detrendLinear.mat", ...
                         "*_ica2_prp_detrendLinear.mat"];
    otherwise
        error("ica_mode must be 'ica', 'ica2', or 'both'");
end

% =========================
% Loop over subjects
% =========================
for si = 1:length(subjects)

    subject = string(subjects{si});

    indir = fullfile( ...
        base_indir, ...
        subject, ...
        "ref_down_filt_chRej", ...
        "epoch_reject", ...
        "ica", ...
        "MAT", ...
        "prp" ...
    );

    fprintf("\n=============================\n");
    fprintf("Subject: %s\n", subject);
    fprintf("Input folder: %s\n", indir);

    files = [];

    for pi = 1:length(file_patterns)
        files = [files; dir(fullfile(indir, file_patterns(pi)))];
    end

    if isempty(files)
        fprintf("No matching PRP files found for %s, skipping.\n", subject);
        continue;
    end

    for fi = 1:length(files)

        infile = fullfile(files(fi).folder, files(fi).name);
        fprintf("\nProcessing %s\n", files(fi).name);

        S = load(infile);
        disp(size(S.eegs));

        % =========================
        % Extract subject ID
        % =========================
        tok = regexp(files(fi).name, "sub-pilot_([A-Za-z0-9]+)", "tokens");

        if isempty(tok)
            tok = regexp(subject, "sub-pilot_([A-Za-z0-9]+)", "tokens");
        end

        if isempty(tok)
            error("Cannot find subject ID for %s", subject);
        end

        subject_short = "sub-" + string(tok{1}{1});

        % =========================
        % Extract session number
        % =========================
        ses_tok = regexp(files(fi).name, "ses-([A-Za-z0-9]+)", "tokens");

        if isempty(ses_tok)
            ses_label = "ses-unknown";
        else
            ses_label = "ses-" + string(ses_tok{1}{1});
        end

        % =========================
        % Extract acquisition number
        % =========================
        acq_tok = regexp(files(fi).name, "acq-([A-Za-z0-9]+)", "tokens");

        if isempty(acq_tok)
            acq_label = "acq-unknown";
        else
            acq_label = "acq-" + string(acq_tok{1}{1});
        end

        % =========================
        % Detect ICA label
        % =========================
        if contains(files(fi).name, "_ica2_")
            ica_label = "ica2";
        elseif contains(files(fi).name, "_ica_")
            ica_label = "ica";
        else
            ica_label = "unknownICA";
        end

        for e = 1:length(emotions_to_save)

            emo = emotions_to_save(e);
            idx = strcmp(string(S.emotions), emo);

            if ~any(idx)
                fprintf("  No %s trials found, skipping.\n", emo);
                continue;
            end

            out = S;

            % Trial-level variables
            out.eegs = S.eegs(idx, :, :);
            out.phonemes = S.phonemes(idx);
            out.emotions = S.emotions(idx);
            out.speech_styles = S.speech_styles(idx);
            out.genders = S.genders(idx);
            out.label_times = S.label_times(idx);

            % Metadata
            out.sub = char(subject_short);
            out.ses = char(ses_label);
            out.acq = char(acq_label);
            out.ica = char(ica_label);

            out.file = char( ...
                subject_short + "_" + ...
                ses_label + "_" + ...
                "task-emo_" + ...
                acq_label + "_" + ...
                ica_label + ...
                "_cond-" + emo + ".mat" ...
            );

            outfile = fullfile(outdir, out.file);

            save(outfile, "-struct", "out", "-v7.3");

            fprintf("  Saved %s %s %s %s: %d PRPs -> %s\n", ...
                ses_label, acq_label, ica_label, emo, sum(idx), outfile);
        end
    end
end

fprintf("\nDone.\n");