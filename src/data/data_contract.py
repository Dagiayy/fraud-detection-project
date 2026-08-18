# src/data/data_contract.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TransactionSchema(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Unique UUID for transaction tracking")
    user_id: int = Field(..., description="Unique user identifier")
    signup_time: str = Field(..., description="ISO datetime string of user registration")
    purchase_time: str = Field(..., description="ISO datetime string of purchase")
    purchase_value: float = Field(..., gt=0, description="Purchase amount (must be positive)")
    age: int = Field(..., ge=18, le=120, description="User age")
    ip_address: int = Field(..., ge=0, description="Integer encoded IP address")
    source: str = Field(..., description="Acquisition channel: Ads, SEO, Direct")
    browser: str = Field(..., description="Browser type: Chrome, FireFox, IE, Safari, Opera")
    sex: str = Field(..., description="Gender: M, F")
    device_id: Optional[str] = Field(default="UNKNOWN", description="Device identifier")
    class_label: Optional[int] = Field(default=None, alias="class", description="0 for Legitimate, 1 for Fraud")

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"Ads", "SEO", "Direct"}
        if v not in allowed:
            return "Direct"
        return v

    @field_validator("browser")
    @classmethod
    def validate_browser(cls, v: str) -> str:
        allowed = {"Chrome", "FireFox", "IE", "Safari", "Opera"}
        if v not in allowed:
            return "Chrome"
        return v

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v.upper() not in {"M", "F"}:
            return "M"
        return v.upper()

class BatchTransactionSchema(BaseModel):
    transactions: List[TransactionSchema]
