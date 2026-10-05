from .platform_motion import (PlatformMotionCurriculum,
                              assert_curriculum_reaches_nominal,
                              episodes_to_reach_nominal,
                              fitted_update_interval)

__all__ = ["PlatformMotionCurriculum", "assert_curriculum_reaches_nominal",
           "episodes_to_reach_nominal", "fitted_update_interval"]
