classdef ExternalStack < handle
    %EXTERNALSTACK Own the DDS agent, Isaac/PX4 and gateway processes.
    %   The external pipeline needs three long-lived processes that are
    %   normally started by hand in three terminals. This class starts them in
    %   the documented order, waits for each readiness signal, and guarantees
    %   that whatever it started is stopped again, including on error.
    %
    %   Processes that are already running are adopted, not restarted, and are
    %   left alone on teardown: a session the user started by hand is theirs.
    %
    %   Example:
    %       s = stack.ExternalStack(cfg);
    %       c = onCleanup(@()delete(s));
    %       s.start();

    properties (SetAccess=private)
        Root            % workspace root (the directory holding scripts/)
        LogDir          % where process logs and pid files are written
        IsaacSimPath    % Isaac Sim release directory
        Headless        % run Isaac without a window
        GatewayArgs     % extra arguments for run_gateway.sh
        Managed         % struct array of processes this object started
        Timeouts        % readiness budgets, seconds
    end

    properties (Constant)
        AgentPort   = 8888   % Micro XRCE-DDS Agent, UDP
        GatewayPort = 14650  % MATLAB <-> gateway, UDP
        % MATLAB prepends its own runtime to the loader path, which breaks
        % ROS 2 and Isaac. Child processes get a clean loader environment.
        EnvPrefix = ['env -u LD_LIBRARY_PATH -u LD_PRELOAD -u QT_PLUGIN_PATH ' ...
            '-u QT_QPA_PLATFORM_PLUGIN_PATH -u GTK_PATH '];
    end

    methods
        function obj = ExternalStack(cfg,varargin)
            p = inputParser;
            p.addParameter('IsaacSimPath',getenv('ISAACSIM_PATH'),@(x)ischar(x)||isstring(x));
            p.addParameter('Headless','auto');
            p.addParameter('GatewayArgs','--target sitl --allow-arm',@(x)ischar(x)||isstring(x));
            p.addParameter('LogDir','',@(x)ischar(x)||isstring(x));
            p.addParameter('AgentTimeout',30,@isnumeric);
            p.addParameter('IsaacTimeout',600,@isnumeric);
            p.addParameter('GatewayTimeout',60,@isnumeric);
            p.parse(varargin{:});
            args = p.Results;

            if ~isfolder(fullfile(cfg.paths.root,'scripts'))
                error('stack:root','No scripts/ directory under %s.',cfg.paths.root);
            end
            obj.Root = cfg.paths.root;
            obj.IsaacSimPath = char(args.IsaacSimPath);
            obj.Headless = stack.ExternalStack.resolveHeadless(args.Headless);
            obj.GatewayArgs = char(args.GatewayArgs);
            if isempty(args.LogDir)
                obj.LogDir = fullfile(tempdir,'ontology_rgat_stack');
            else
                obj.LogDir = char(args.LogDir);
            end
            if ~isfolder(obj.LogDir), mkdir(obj.LogDir); end
            obj.Timeouts = struct('agent',args.AgentTimeout, ...
                'isaac',args.IsaacTimeout,'gateway',args.GatewayTimeout);
            obj.Managed = struct('name',{},'pid',{},'log',{});
        end

        function start(obj)
            %START Bring the stack up in the order docs/OPERATIONS.md requires.
            obj.startAgent();
            obj.startIsaac();
            obj.startGateway();
            fprintf('External stack ready (logs in %s).\n',obj.LogDir);
        end

        function startAgent(obj)
            if stack.ExternalStack.udpPortBound(obj.AgentPort)
                fprintf('DDS agent already listening on UDP %d; adopting it.\n',obj.AgentPort);
                return;
            end
            log = obj.launch('agent',obj.script('run_dds_agent.sh'));
            obj.waitFor(@()stack.ExternalStack.udpPortBound(obj.AgentPort), ...
                obj.Timeouts.agent,'DDS agent',log);
            fprintf('DDS agent listening on UDP %d.\n',obj.AgentPort);
        end

        function startIsaac(obj)
            if stack.ExternalStack.processRunning('landing_world.py')
                fprintf('Isaac/PX4 already running; adopting it.\n');
                return;
            end
            if isempty(obj.IsaacSimPath) || ~isfile(fullfile(obj.IsaacSimPath,'python.sh'))
                error('stack:isaacPath', ...
                    ['Isaac Sim not found. Pass IsaacSimPath or set ISAACSIM_PATH ' ...
                    'to the directory containing python.sh (got "%s").'],obj.IsaacSimPath);
            end
            environment = {'ISAACSIM_PATH',obj.IsaacSimPath; ...
                'HEADLESS',num2str(double(obj.Headless))};
            if ~obj.Headless
                % The launcher runs detached, so the window needs its display
                % named explicitly rather than inherited from a terminal.
                display = getenv('DISPLAY');
                if isempty(display)
                    error('stack:noDisplay', ...
                        ['Isaac Sim was asked for a window but DISPLAY is not ' ...
                        'set. Run MATLAB from a graphical session, or pass ' ...
                        '''Headless'',true.']);
                end
                environment(end+1,:) = {'DISPLAY',display};
                xauth = getenv('XAUTHORITY');
                if ~isempty(xauth)
                    environment(end+1,:) = {'XAUTHORITY',xauth};
                end
                fprintf('Isaac Sim will open a window on DISPLAY %s.\n',display);
            end
            log = obj.launch('isaac',obj.script('run_isaac.sh'),environment);
            % PX4 is launched by Pegasus once the world is loaded, so its
            % banner is the only signal that the whole simulator is up.
            fprintf('Waiting for Isaac Sim and PX4 (first boot can take minutes)...\n');
            obj.waitFor(@()stack.ExternalStack.logContains(log,'Ready for takeoff'), ...
                obj.Timeouts.isaac,'Isaac Sim + PX4 SITL',log);
            fprintf('Isaac Sim up and PX4 reports Ready for takeoff.\n');
        end

        function startGateway(obj)
            if stack.ExternalStack.udpPortBound(obj.GatewayPort)
                fprintf('Gateway already listening on UDP %d; adopting it.\n',obj.GatewayPort);
                return;
            end
            command = sprintf('%s %s',obj.script('run_gateway.sh'),obj.GatewayArgs);
            log = obj.launch('gateway',command);
            obj.waitFor(@()stack.ExternalStack.udpPortBound(obj.GatewayPort), ...
                obj.Timeouts.gateway,'PX4 gateway',log);
            fprintf('Gateway listening on UDP %d.\n',obj.GatewayPort);
        end

        function restart(obj)
            %RESTART Cycle the simulator. PX4 SITL degrades over long sessions
            %   (battery_status goes stale under lockstep and arming is then
            %   refused), so a long sweep may need a clean simulator.
            obj.stop();
            pause(3);
            obj.start();
        end

        function stop(obj)
            %STOP Terminate only the processes this object started.
            for k = numel(obj.Managed):-1:1
                entry = obj.Managed(k);
                fprintf('Stopping %s (pid %d).\n',entry.name,entry.pid);
                stack.ExternalStack.killGroup(entry.pid);
                obj.Managed(k) = [];
            end
        end

        function delete(obj)
            try
                obj.stop();
            catch err
                warning('stack:teardown','Stack teardown failed: %s',err.message);
            end
        end

        function tf = isReady(obj)
            tf = stack.ExternalStack.udpPortBound(obj.AgentPort) && ...
                stack.ExternalStack.udpPortBound(obj.GatewayPort) && ...
                stack.ExternalStack.processRunning('landing_world.py');
        end
    end

    methods (Access=private)
        function path = script(obj,name)
            path = stack.ExternalStack.quote(fullfile(obj.Root,'scripts',name));
        end

        function log = launch(obj,name,command,environment)
            %LAUNCH Start one process in its own session and record its pid.
            %   ENVIRONMENT is an N-by-2 cell array of name/value pairs. They
            %   are exported on their own lines: a "VAR=value cmd" prefix is
            %   not a thing `exec` understands.
            if nargin < 4, environment = cell(0,2); end
            log = fullfile(obj.LogDir,[name '.log']);
            pidFile = fullfile(obj.LogDir,[name '.pid']);
            if isfile(pidFile), delete(pidFile); end
            % A generated launcher avoids nested shell quoting, which matters
            % because the workspace path is not ASCII.
            launcher = fullfile(obj.LogDir,[name '.sh']);
            fid = fopen(launcher,'w','n','UTF-8');
            if fid < 0, error('stack:launcher','Cannot write %s.',launcher); end
            fprintf(fid,'#!/usr/bin/env bash\n');
            fprintf(fid,'echo $$ > %s\n',stack.ExternalStack.quote(pidFile));
            for k = 1:size(environment,1)
                fprintf(fid,'export %s=%s\n',environment{k,1}, ...
                    stack.ExternalStack.quote(environment{k,2}));
            end
            fprintf(fid,'exec %s\n',command);
            fclose(fid);
            % setsid puts the whole tree in one process group, so Pegasus'
            % PX4 child dies with Isaac instead of holding TCP 4560.
            system(sprintf('chmod +x %s',stack.ExternalStack.quote(launcher)));
            system(sprintf('%ssetsid %s > %s 2>&1 < /dev/null &', ...
                obj.EnvPrefix,stack.ExternalStack.quote(launcher), ...
                stack.ExternalStack.quote(log)));
            pid = obj.readPid(pidFile,10);
            obj.Managed(end+1) = struct('name',name,'pid',pid,'log',log);
        end

        function pid = readPid(~,pidFile,timeout)
            t0 = tic;
            while toc(t0) < timeout
                if isfile(pidFile)
                    text = strtrim(fileread(pidFile));
                    pid = str2double(text);
                    if isfinite(pid) && pid > 0, return; end
                end
                pause(0.1);
            end
            error('stack:pid','Process did not report a pid via %s.',pidFile);
        end

        function waitFor(obj,predicate,timeout,description,log)
            t0 = tic;
            while toc(t0) < timeout
                if predicate(), return; end
                if ~obj.allManagedAlive()
                    error('stack:died','%s exited during startup. Log:\n%s', ...
                        description,stack.ExternalStack.tailLog(log));
                end
                pause(0.5);
            end
            error('stack:timeout','%s was not ready within %.0f s. Log:\n%s', ...
                description,timeout,stack.ExternalStack.tailLog(log));
        end

        function tf = allManagedAlive(obj)
            tf = true;
            for k = 1:numel(obj.Managed)
                if system(sprintf('kill -0 %d 2>/dev/null',obj.Managed(k).pid)) ~= 0
                    tf = false; return;
                end
            end
        end
    end

    methods (Static)
        function text = quote(value)
            text = ['"' strrep(char(value),'"','\"') '"'];
        end

        function tf = resolveHeadless(value)
            %RESOLVEHEADLESS 'auto' shows a window wherever there is a display.
            if (islogical(value) || isnumeric(value)) && isscalar(value)
                tf = logical(value);
                return;
            end
            if (ischar(value) || isstring(value)) && strcmpi(value,'auto')
                tf = isempty(getenv('DISPLAY'));
                return;
            end
            error('stack:headless', ...
                ['Headless must be true, false or ''auto'' (got a %s). ' ...
                'If this looks impossible, MATLAB is holding an older copy ' ...
                'of this class: run "clear classes" and try again.'],class(value));
        end

        function tf = udpPortBound(port)
            status = system(sprintf('ss -lnuH 2>/dev/null | grep -q ":%d "',port));
            tf = status == 0;
        end

        function tf = processRunning(pattern)
            % The bracket keeps the grep from matching its own command line.
            guarded = ['[' pattern(1) ']' pattern(2:end)];
            status = system(sprintf('ps -eo cmd | grep -q -- %s', ...
                stack.ExternalStack.quote(guarded)));
            tf = status == 0;
        end

        function tf = logContains(logFile,needle)
            tf = false;
            if ~isfile(logFile), return; end
            status = system(sprintf('grep -q -- %s %s', ...
                stack.ExternalStack.quote(needle),stack.ExternalStack.quote(logFile)));
            tf = status == 0;
        end

        function text = tailLog(logFile)
            text = '(no log)';
            if isempty(logFile) || ~isfile(logFile), return; end
            [~,text] = system(sprintf('tail -n 20 %s', ...
                stack.ExternalStack.quote(logFile)));
        end

        function killGroup(pid)
            % Negative pid targets the process group created by setsid, so
            % Pegasus' PX4 child goes down with the simulator.
            system(sprintf('kill -TERM -%d 2>/dev/null',pid));
            for k = 1:40
                if system(sprintf('kill -0 %d 2>/dev/null',pid)) ~= 0, break; end
                pause(0.25);
            end
            system(sprintf('kill -KILL -%d 2>/dev/null',pid));
        end
    end
end
