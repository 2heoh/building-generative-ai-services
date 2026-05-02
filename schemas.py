# schemas.py

from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    PositiveInt,
    validate_call,
    IPvAnyAddress,
)
# class ModelRequest(BaseModel):
#     prompt: Annotated[str, Field(min_length=1, max_length=10000)] 

class ModelRequest(BaseModel):
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]

class ModelResponse(BaseModel):
    request_id: Annotated[str, Field(default_factory=lambda: uuid4().hex)] 
    # no defaults set for ip field
    # raise ValidationError if a valid IP address or None is not provided
    ip: Annotated[str, IPvAnyAddress] | None 
    content: Annotated[str | None, Field(min_length=0, max_length=10000)] 
    created_at: datetime = datetime.now()

@dataclass
class TextModelRequest: 
    model: Literal["tinyLlama", "gemma2b"]
    prompt: str
    temperature: float

# @dataclass
class TextModelResponse(BaseModel): 
    # response: str
    tokens: Annotated[int, Field(ge=0)]

ImageSize = Annotated[
    tuple[PositiveInt, PositiveInt], "Width and height of an image in pixels"
]
SupportedModels = Annotated[
    Literal["tinysd", "sd1.5"], "Supported Image Generation Models"
]

@validate_call 
def is_square_image(value: ImageSize) -> ImageSize: 
    if value[0] / value[1] != 1:
        raise ValueError("Only square images are supported")
    if value[0] not in [512, 1024]:
        raise ValueError(f"Invalid output size: {value} - expected 512 or 1024")
    return value

@validate_call 
def is_valid_inference_step(
    num_inference_steps: int, model: SupportedModels
) -> int:
    if model == "tinysd" and num_inference_steps > 2000: 
        raise ValueError(
            "TinySD model cannot have more than 2000 inference steps"
        )
    return num_inference_steps

OutputSize = Annotated[ImageSize, AfterValidator(is_square_image)] 
InferenceSteps = Annotated[ 
    int,
    AfterValidator(
        lambda v, values: is_valid_inference_step(v, values["model"])
    ),
]

class ImageModelRequest(ModelRequest):
    model: SupportedModels
    output_size: OutputSize 
    num_inference_steps: InferenceSteps = 200

# class ImageModelRequest(ModelRequest):
#     model: Literal["tinysd", "sd1.5"]
#     output_size: ImageSize 
#     num_inference_steps: Annotated[int, Field(ge=0, le=2000)] = 200 

class ImageModelResponse(ModelResponse):
    size: ImageSize 
    url: Annotated[str, HttpUrl] | None = None