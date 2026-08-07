from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class MachineData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"]

    air_temperature: float = Field(ge=0)
    process_temperature: float = Field(ge=0)
    rotational_speed: float = Field(ge=0)
    torque: float = Field(ge=0)
    tool_wear: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    failure_probability: float = Field(ge=0, le=1)