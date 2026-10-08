function fig = view_results(csvPath)
% VIEW_RESULTS Minimal viewer for a matlab-port evaluation CSV.
T=readtable(csvPath);
required={'time_s','horizontal_distance_m','height_m','uav_x_m','uav_z_m', ...
    'pad_x_m','pad_z_m','requested_ax','requested_az','applied_ax','applied_az'};
assert(all(ismember(required,T.Properties.VariableNames)), ...
    'matlab_port:ViewerSchema','CSV does not contain the matlab-port viewer schema.');
fig=uifigure('Name','MATLAB direct-policy result');
tabs=uitabgroup(fig);
trajectory=uitab(tabs,'Title','Trajectory');
grid=uigridlayout(trajectory,[1,2]);
left=uiaxes(grid); right=uiaxes(grid);
plot(left,T.time_s,T.horizontal_distance_m,'DisplayName','horizontal'); hold(left,'on');
plot(left,T.time_s,T.height_m,'DisplayName','height'); legend(left,'show'); grid(left,'on');
xlabel(left,'time [s]'); ylabel(left,'relative distance [m]');
plot(right,T.uav_x_m,T.uav_z_m,'DisplayName','UAV'); hold(right,'on');
plot(right,T.pad_x_m,T.pad_z_m,'DisplayName','UGV/pad'); legend(right,'show'); grid(right,'on');
xlabel(right,'x [m]'); ylabel(right,'z [m]');
authority=uitab(tabs,'Title','Control authority');
axes2=uiaxes(authority);
plot(axes2,T.time_s,T.requested_ax,'--',T.time_s,T.applied_ax,'-', ...
    T.time_s,T.requested_az,'--',T.time_s,T.applied_az,'-');
legend(axes2,{'requested ax','applied ax','requested az','applied az'}); grid(axes2,'on');
xlabel(axes2,'time [s]'); ylabel(axes2,'acceleration [m/s^2]');
end
