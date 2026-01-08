"""Therapist agent implementations for therapy simulation."""

from .omniscient_therapist import OmniscientStrategicTherapist as OmniscientStrategicTherapistV1
from .omniscient_therapist_v2 import OmniscientStrategicTherapist as OmniscientStrategicTherapistV2

# Default to v2 for backward compatibility (has all v1 features + feedback monitoring)
OmniscientStrategicTherapist = OmniscientStrategicTherapistV2

__all__ = [
    'OmniscientStrategicTherapist',  # v2 by default
    'OmniscientStrategicTherapistV1',
    'OmniscientStrategicTherapistV2'
]
