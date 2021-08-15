from typing import List

from pydantic import BaseModel


class InferenceResponse(BaseModel):
    classification: str
    confidence: float
    inference: List[float]
    time_to_complete: float

    @staticmethod
    def asJson(classification, confidence, inference, time_to_complete):
        return {
            "classification": classification,
            "confidence": confidence,
            "inference": inference,
            "time_to_complete": time_to_complete
        }
