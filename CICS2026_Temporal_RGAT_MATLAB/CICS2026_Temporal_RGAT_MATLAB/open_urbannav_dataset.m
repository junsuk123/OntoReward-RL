function open_urbannav_dataset()
%OPEN_URBANNAV_DATASET Open the official UrbanNav-HK dataset repository.
url='https://github.com/IPNL-POLYU/UrbanNavDataset';
fprintf('Opening official UrbanNav dataset page:\n%s\n',url);
web(url,'-browser');
fprintf(['\nDownload GNSS + Ground Truth for Medium, Harsh, and Deep Urban from the official page.\n' ...
    'Full ROS datasets are very large and are intentionally not bundled here.\n' ...
    'For strict prior-study reproduction, the original processed 12-feature tables are preferable.\n']);
end
