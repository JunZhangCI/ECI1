function [comp, fdone] = load_existing_csv(csv_name)
    comp = [];
    fdone = [];
    if exist(csv_name, 'file')
        try
            comp = readtable(csv_name);
            fdone = unique(comp.filename);
        catch
            comp = [];
            fdone = [];
        end
    end
end