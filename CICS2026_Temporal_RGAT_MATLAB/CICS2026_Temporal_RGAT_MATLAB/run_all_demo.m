function run_all_demo()
%RUN_ALL_DEMO End-to-end software smoke test using synthetic data.
root=fileparts(mfilename('fullpath'));
addpath(root); addpath(fullfile(root,'tests'));
check_environment();
if ~isfile(fullfile(root,'data','demo','medium.csv'))
    generate_demo_datasets();
end
run_unit_tests();
fprintf('\n*** DEMO MODE: synthetic data. DO NOT cite metrics in a paper. ***\n');
run_benchmark("demo");
end
