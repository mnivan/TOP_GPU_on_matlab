function runDir = latestCompletedRun(prefix,requiredFile)
% Find the newest completed TOP_GPU output matching a reproduction case.
root = fullfile(fileparts(fileparts(mfilename('fullpath'))),'out');
categories = {'mtop','pio','verification','benchmark','memory','validation'};
roots = [{root}, cellfun(@(c) fullfile(root,c),categories,'UniformOutput',false)];
matches = [];
for k = 1:numel(roots)
    matches = [matches; dir(fullfile(roots{k},[prefix '*']))]; %#ok<AGROW>
end
matches = matches([matches.isdir]);
pattern = ['^' regexptranslate('escape',prefix) '(_t[0-9]+)?$'];
matches = matches(~cellfun('isempty',regexp({matches.name},pattern,'once')));
if isempty(matches)
    error('reproduce:MissingRun','No output directory starts with %s.',prefix);
end
[~,order] = sort([matches.datenum],'descend');
for index = order
    candidate = fullfile(matches(index).folder,matches(index).name);
    if isfile(fullfile(candidate,requiredFile))
        runDir = candidate;
        return;
    end
end
error('reproduce:MissingOutput','No %s found for %s.',requiredFile,prefix);
end
