"""Simple state machine for drone mission sequencing.

Each Step represents one phase of the mission (takeoff, hover, waypoint, land, etc.).
Steps have conditions to check completion and connections to possible next steps.
The StateMachine advances through steps based on their conditions.
"""
from dronePositionCTRL import DronePositionCTRL
import numpy as np


class Step:
    def __init__(self, idx=0, name="", connections=None, action=None, is_done=None, on_enter=None, on_exit=None, timeout=None):
        self.idx = idx
        self.name = name or f"step_{idx}"
        self.connections = connections or []
        self.action = action
        self.is_done = is_done
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.timeout = timeout

        # Runtime state
        self.entered_at = None
        self.tick_count = 0

    def enter(self, ctx, sim_time):
        """Called once when state machine enters this step."""
        self.entered_at = sim_time
        self.tick_count = 0
        if self.on_enter is not None: self.on_enter(ctx)

    def exit(self, ctx):
        if self.on_exit is not None: self.on_exit(ctx)

    def run(self, ctx, sim_time):
        self.tick_count += 1

        if self.action is not None: self.action(ctx)

        # Check timeout
        if self.timeout is not None and self.entered_at is not None:
            if sim_time - self.entered_at > self.timeout:
                # Pick first connection as default exit, or stop if none
                return self.connections[0] if self.connections else -1

        # Check completion condition
        if self.is_done is not None:
            next_idx = self.is_done(ctx)
            
            if  next_idx is not None and (next_idx in self.connections or not self.connections):
                return next_idx
                
        return None

    def time_in_step(self, sim_time):
        if self.entered_at is None: return 0.0
        return sim_time - self.entered_at

    def __repr__(self):
        return f"Step({self.idx}, {self.name!r}, conns={self.connections})"


class StateMachine:
    """Linear or branching sequence of Steps.
    
    Owns a dict of steps keyed by idx, tracks current step, drives transitions.
    """

    def __init__(self, steps=None, start_idx=0, ctx=None):
        """
        steps     : list of Step objects, or dict {idx: Step}
        start_idx : idx of the first step
        ctx       : context object passed to all step callbacks (e.g. the drone controller)
        """
        if isinstance(steps, dict):
            self.steps = steps
        else:
            self.steps = {s.idx: s for s in (steps or [])}

        self.ctx = ctx
        self.current_idx = start_idx
        self.previous_idx = None
        self.history = []           # list of (sim_time, from_idx, to_idx) transitions
        self.finished = False
        self._entered_initial = False

    @property
    def current(self):
        """The currently-active Step object."""
        return self.steps.get(self.current_idx)

    def add_step(self, step):
        """Add a step to the machine."""
        self.steps[step.idx] = step

    def tick(self, sim_time):
        """
        Advance one tick. Runs current step's action, checks for transition.
        Returns the current step's idx (or -1 if finished).
        """
        if self.finished:
            return -1

        step = self.current
        if step is None:
            self.finished = True
            return -1

        # First tick in a new step — call on_enter
        if not self._entered_initial:
            step.enter(self.ctx, sim_time)
            self._entered_initial = True

        # Run the step
        next_idx = step.run(self.ctx, sim_time)

        # Transition if requested
        if next_idx is not None:
            self._transition(next_idx, sim_time)

        return self.current_idx if not self.finished else -1

    def _transition(self, next_idx, sim_time):
        """Move to a new step."""
        current = self.current
        if current is not None:
            current.exit(self.ctx)

        self.history.append((sim_time, self.current_idx, next_idx))
        self.previous_idx = self.current_idx
        self.current_idx = next_idx
        self._entered_initial = False

        if next_idx == -1 or next_idx not in self.steps:
            self.finished = True
        else:
            # Enter the new step
            self.steps[next_idx].enter(self.ctx, sim_time)
            self._entered_initial = True

    def jump_to(self, idx, sim_time):
        """Force transition to a specific step (bypasses connection check)."""
        self._transition(idx, sim_time)

    def reset(self, start_idx=None):
        """Reset to the beginning (or a specific step)."""
        if start_idx is not None:
            self.current_idx = start_idx
        self.previous_idx = None
        self.history = []
        self.finished = False
        self._entered_initial = False

    def __repr__(self):
        return (f"StateMachine(current={self.current_idx}, "
                f"finished={self.finished}, steps={len(self.steps)})")