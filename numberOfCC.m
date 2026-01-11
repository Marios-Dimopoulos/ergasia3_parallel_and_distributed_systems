try 
    fprintf('Loading matrix...\n');
    homeDir = getenv('HOME');
    filePath = fullfile(homeDir, 'ergasia3_parallhla', 'matrices', 'road_central.mat');
    
    data = load(filePath, 'Problem');
    A = data.Problem.A;

    fprintf('Computing connected components...\n');
    num_components = max(conncomp(graph(A)));

    fprintf('TOTAL CONNECTED COMPONENTS: %d\n', num_components);
catch ME
    fprintf('An error occurred: %s\n', ME.message);
end
exit;