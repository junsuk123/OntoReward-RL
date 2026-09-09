classdef PX4Bridge < handle
    %PX4BRIDGE Versioned MATLAB UDP client for the external PX4 gateway.
    properties (SetAccess=private)
        Socket
        Config
        Sequence uint64 = uint64(0)
        LastState = struct()
        Expected = struct()
        ControlPeriodUs double = 0
        LastPx4TimeUs = []
    end
    methods
        function obj=PX4Bridge(cfg)
            if ~isfield(cfg,'external') || ~cfg.external.enabled
                error('External simulation configuration is required.');
            end
            obj.Config=cfg.external;
            obj.Expected=struct('collectiveSpan',cfg.rl.collectiveSpan, ...
                'maxRollPitch',cfg.rl.maxRollPitch,'maxYawRate',cfg.rl.maxYawRate);
            obj.ControlPeriodUs=round(cfg.sim.dt*1e6);
            obj.Socket=udpport('datagram','IPV4','LocalHost',obj.Config.localHost, ...
                'LocalPort',obj.Config.localPort,'Timeout',obj.Config.timeout);
            flush(obj.Socket);
            hello=obj.transact('hello',struct(),{'state'});
            obj.assertControlMapping(hello);
        end

        function assertControlMapping(obj,state)
            %ASSERTCONTROLMAPPING Refuse a gateway that scales actions differently.
            % The policy emits normalised actions against cfg.rl.*; if the
            % gateway converts them with different limits the vehicle is
            % silently under- or over-actuated and every result is invalid.
            if ~isfield(state,'extra') || ~isstruct(state.extra) || ...
                    ~isfield(state.extra,'control_mapping')
                warning('PX4Bridge:mappingUnreported', ...
                    'Gateway does not report its control mapping; cannot verify it.');
                return;
            end
            m=state.extra.control_mapping;
            checks={'collective_span',obj.Expected.collectiveSpan,'cfg.rl.collectiveSpan'; ...
                'max_roll_pitch_rad',obj.Expected.maxRollPitch,'cfg.rl.maxRollPitch'; ...
                'max_yaw_rate_rad_s',obj.Expected.maxYawRate,'cfg.rl.maxYawRate'};
            for k=1:size(checks,1)
                field=checks{k,1};
                if ~isfield(m,field), continue; end
                if abs(double(m.(field))-checks{k,2})>1e-6
                    error(['Gateway %s is %.4f but %s is %.4f. Fix config/system.yaml ' ...
                        'so the action scaling matches the policy.'], ...
                        field,double(m.(field)),checks{k,3},checks{k,2});
                end
            end
        end

        function state=reset(obj,seed)
            ack=obj.transact('reset',struct('seed',double(seed), ...
                'wind_scale',double(obj.Config.windScale), ...
                'pad_scale',double(obj.Config.padScale)),{'ack'});
            pause(obj.Config.resetSettle);
            state=obj.waitValidState();
            if ~obj.Config.autoArm
                return;
            end
            % The entry pose is flown by PX4, never teleported: Pegasus cannot
            % reset the PX4 estimator, so a jump would leave the policy reading
            % a diverged EKF for the whole episode. It is an offset in the pad
            % frame, and the gateway re-aims it at the live deck every control
            % tick, so PX4 chases a moving entry point instead of holding a
            % point the rover has already driven away from.
            entry=obj.entryPose(ack);
            obj.transact('goto',struct('position',entry.position.', ...
                'yaw',entry.yaw,'frame',entry.frame, ...
                'hold_s',obj.Config.entryTimeout),{'ack'});
            % Let the setpoint stream establish offboard before arming.
            pause(obj.Config.prestreamCount/obj.Config.controlHz);
            state=obj.waitAtEntry(entry.position);
            % The episode clock starts at handover, not at the reset.
            obj.LastPx4TimeUs=double(state.px4_time_us);
        end

        function entry=entryPose(obj,ack)
            %ENTRYPOSE The seeded entry point, as an offset from the deck.
            frame=obj.Config.entryFrame;
            if ~isfield(ack,'detail') || ~isstruct(ack.detail)
                error(['Reset acknowledgement carries no entry pose. Restart ' ...
                    'Isaac with the current landing_world.py.']);
            end
            detail=ack.detail;
            if strcmpi(frame,'pad')
                if ~isfield(detail,'entry_offset_pad_m')
                    error(['Reset acknowledgement carries no pad-relative entry ' ...
                        'offset. Isaac is running a landing_world.py from before ' ...
                        'the pad was put on a rover; restart it.']);
                end
                position=double(detail.entry_offset_pad_m(:));
            else
                if ~isfield(detail,'entry_position_enu_m')
                    error('Reset acknowledgement carries no entry position.');
                end
                position=double(detail.entry_position_enu_m(:));
            end
            entry=struct('position',position,'yaw',0,'frame',lower(char(frame)));
            if isfield(detail,'entry_yaw_enu_rad')
                entry.yaw=double(detail.entry_yaw_enu_rad);
            end
            if numel(entry.position)~=3 || any(~isfinite(entry.position))
                error('Reset acknowledgement carries a malformed entry position.');
            end
        end

        function state=waitAtEntry(obj,target)
            %WAITATENTRY Hand over only once PX4 holds the entry pose.
            t0=tic; settledSince=[]; state=[]; lastArm=-inf;
            while toc(t0)<obj.Config.entryTimeout
                try
                    state=obj.getState();
                    if ~state.armed && toc(t0)-lastArm>=obj.Config.armRetry
                        % PX4 rejects arming in transient pre-flight states, so
                        % one request is not enough to start the climb.
                        lastArm=toc(t0);
                        obj.transact('arm',struct(),{'ack'});
                    end
                catch
                    % A brief estimator or link transient during the climb is
                    % not a handover failure; only the deadline decides.
                    settledSince=[]; pause(0.05); continue;
                end
                atTarget=norm(state.position-target)<=obj.Config.entryTolerance && ...
                    norm(state.velocity)<=obj.Config.entrySpeedTolerance;
                if ~atTarget
                    settledSince=[];
                elseif isempty(settledSince)
                    settledSince=tic;
                elseif toc(settledSince)>=obj.Config.entrySettle
                    return;
                end
                pause(0.02);
            end
            if isempty(state)
                error('PX4 published no state while climbing to the entry pose.');
            end
            error(['PX4 did not hold the entry pose within %.1f s ' ...
                '(offset %.2f m, speed %.2f m/s).'],obj.Config.entryTimeout, ...
                norm(state.position-target),norm(state.velocity));
        end

        function state=step(obj,action)
            action=double(action(:));
            if numel(action)~=4 || any(~isfinite(action)) || any(abs(action)>1)
                error('Action must contain four finite values in [-1,1].');
            end
            reply=obj.transact('action',struct('action',action.'),{'state'});
            state=obj.paceToControlPeriod(obj.validateState(reply));
            obj.LastState=state;
        end

        function state=paceToControlPeriod(obj,state)
            %PACETOCONTROLPERIOD Let one control period of simulated time pass.
            % The gateway answers as soon as PX4 publishes odometry, which is
            % several times faster than the control rate. Returning that sample
            % straight away ran the policy far faster than cfg.sim.dt while the
            % episode clock still advanced cfg.sim.dt per step, so an episode
            % covered a fraction of its nominal duration and the vehicle ran out
            % of steps before it could land. PX4's clock is the simulator's
            % clock under lockstep, so pace on that and never on wall time.
            if obj.ControlPeriodUs<=0 || ~isfield(state,'px4_time_us'), return; end
            if isempty(obj.LastPx4TimeUs)
                obj.LastPx4TimeUs=double(state.px4_time_us);
                return;
            end
            deadline=obj.LastPx4TimeUs+obj.ControlPeriodUs;
            t0=tic;
            while double(state.px4_time_us)<deadline
                if toc(t0)>obj.Config.timeout
                    error(['PX4 simulated time advanced only %.1f ms in %.2f s ' ...
                        'of wall time; the simulator has stalled.'], ...
                        (double(state.px4_time_us)-obj.LastPx4TimeUs)/1e3, ...
                        obj.Config.timeout);
                end
                state=obj.validateState(obj.transact('state',struct(),{'state'}));
            end
            % Fixed cadence: measure the next period from the deadline so that
            % sampling jitter cannot accumulate into a drifting episode clock.
            obj.LastPx4TimeUs=deadline;
        end

        function state=getState(obj)
            reply=obj.transact('state',struct(),{'state'});
            state=obj.validateState(reply);
            obj.LastState=state;
        end

        function disarm(obj)
            obj.transact('disarm',struct(),{'ack'});
        end

        function enableOffboard(obj)
            obj.transact('enable_offboard',struct(),{'ack'});
        end

        function disableOffboard(obj)
            obj.transact('disable_offboard',struct(),{'ack'});
        end

        function state=waitValidState(obj)
            t0=tic; lastError=[];
            while toc(t0)<obj.Config.estimatorWarmup
                try
                    state=obj.getState(); return;
                catch ME
                    lastError=ME; pause(0.05);
                end
            end
            if isempty(lastError)
                error('PX4 estimator did not publish state within %.1f s.', ...
                    obj.Config.estimatorWarmup);
            end
            rethrow(lastError);
        end

        function delete(obj)
            try
                if strcmpi(obj.Config.target,'sitl'), obj.disarm(); end
            catch
            end
            % Dropping the reference is not enough: udpport holds the socket
            % until it is deleted, and a sweep that builds one bridge per
            % episode would run out of the local port on the second episode.
            try
                if ~isempty(obj.Socket) && isvalid(obj.Socket)
                    delete(obj.Socket);
                end
            catch
            end
            obj.Socket=[];
        end
    end

    methods (Access=private)
        function reply=transact(obj,type,fields,expected)
            obj.Sequence=obj.Sequence+1;
            seq=double(obj.Sequence);
            message=fields; message.v=obj.Config.protocolVersion;
            message.type=type; message.seq=seq;
            message.time_ns=double(posixtime(datetime('now'))*1e9);
            payload=unicode2native(jsonencode(message),'UTF-8');
            write(obj.Socket,uint8(payload),obj.Config.gatewayHost,obj.Config.gatewayPort);
            t0=tic;
            while toc(t0)<obj.Config.timeout
                if obj.Socket.NumDatagramsAvailable==0
                    pause(0.002); continue;
                end
                response=obj.decodeDatagram(read(obj.Socket,1,'uint8'));
                if isfield(response,'type') && strcmp(response.type,'error')
                    error('PX4 gateway rejected request: %s',response.error);
                end
                if isfield(response,'ack_seq') && response.ack_seq==seq && ...
                        any(strcmp(response.type,expected))
                    reply=response; return;
                end
            end
            error('PX4 gateway timeout after %.2f s (%s seq=%d).', ...
                obj.Config.timeout,type,seq);
        end

        function response=decodeDatagram(~,datagram)
            if isstruct(datagram) && isfield(datagram,'Data')
                raw=datagram(1).Data;
            elseif istable(datagram) && any(strcmp(datagram.Properties.VariableNames,'Data'))
                raw=datagram.Data{1};
            elseif isobject(datagram) && isprop(datagram,'Data')
                raw=datagram(1).Data;
            else
                raw=datagram;
            end
            if iscell(raw), raw=raw{1}; end
            response=jsondecode(native2unicode(uint8(raw(:).'),'UTF-8'));
        end

        function state=validateState(~,state)
            required={'position','velocity','quaternion_wxyz','angular_velocity', ...
                'acceleration','wind','aero_force','marker_quality','estimator_valid', ...
                'pad','battery'};
            for k=1:numel(required)
                if ~isfield(state,required{k})
                    error(['Gateway state is missing field %s. A gateway from ' ...
                        'before the moving pad and the energy budget cannot be ' ...
                        'used with this adapter.'],required{k});
                end
            end
            % The target moves, so a gateway that still reports world-frame
            % position would silently be asking the policy to land on the origin.
            if isfield(state,'position_frame') && ~strcmp(state.position_frame,'pad')
                error('Gateway reports %s-frame position; this adapter needs pad-frame.', ...
                    state.position_frame);
            end
            if ~isfield(state,'position_frame')
                error(['Gateway does not declare its position frame. Restart it ' ...
                    'from the current ros2_gateway.py.']);
            end
            numericFields={'position','velocity','quaternion_wxyz','angular_velocity', ...
                'acceleration','wind','aero_force','marker_quality'};
            for k=1:numel(numericFields)
                if any(~isfinite(double(state.(numericFields{k})(:))))
                    error('Gateway state contains non-finite %s.',numericFields{k});
                end
            end
            if ~state.estimator_valid
                error('PX4 estimator state is not valid yet.');
            end
            if ~strcmp(state.frame,'ENU_FLU')
                error('Unsupported gateway frame: %s',state.frame);
            end
            state.position=double(state.position(:));
            state.velocity=double(state.velocity(:));
            state.quaternion_wxyz=double(state.quaternion_wxyz(:));
            state.angular_velocity=double(state.angular_velocity(:));
            state.acceleration=double(state.acceleration(:));
            state.wind=double(state.wind(:));
            state.aero_force=double(state.aero_force(:));
            state.marker_quality=double(state.marker_quality);
        end
    end
end
