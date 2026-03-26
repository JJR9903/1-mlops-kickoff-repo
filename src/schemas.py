from typing import Optional
from pydantic import BaseModel, ConfigDict


class TelcoChurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: float
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictResponse(BaseModel):
    prediction: int
    proba: Optional[float] = None
