from typing import List
from pydantic import BaseModel
import numpy as np


class InferenceRequest(BaseModel):
    data: List[List[float]]

    @property
    def get_data(self):
        return np.array(self.data)
