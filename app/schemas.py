from pydantic import BaseModel, Field


class DiabetesFeatures(BaseModel):
    pregnancies: float = Field(ge=0)
    glucose: float = Field(ge=0)
    blood_pressure: float = Field(ge=0)
    skin_thickness: float = Field(ge=0)
    insulin: float = Field(ge=0)
    bmi: float = Field(ge=0)
    diabetes_pedigree_function: float = Field(ge=0)
    age: float = Field(ge=0)


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
