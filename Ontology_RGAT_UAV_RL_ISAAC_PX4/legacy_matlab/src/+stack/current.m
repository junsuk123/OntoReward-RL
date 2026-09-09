function held = current(newStack)
%CURRENT The stack the running pipeline owns, or [] when there is none.
%   sim.resetState consults this to recover from a simulator that has stopped
%   accepting arm commands, which PX4 SITL does after long sessions. Only
%   run_pipeline registers a stack; a hand-started stack is never restarted
%   from under the user.
%
%   stack.current(s)   register s
%   stack.current([])  clear the registration
%   s = stack.current()

persistent registered
if nargin > 0
    if isempty(newStack)
        registered = [];
    else
        registered = newStack;
    end
end
if isempty(registered) || ~isvalid(registered)
    held = [];
else
    held = registered;
end
end
