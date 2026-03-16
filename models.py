from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean
from database import Base
import datetime

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    tracking_id = Column(String, unique=True, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    address = Column(Text)
    category = Column(String)
    department = Column(String)
    document_name = Column(String)
    document_desc = Column(Text)
    ocr_text = Column(Text)
    status = Column(String, default="In Review")
    is_verified = Column(String, default="Pending") # Verified, Rejected, Pending
    ai_priority = Column(Integer, default=3) # 1-5
    ai_reasoning = Column(Text)
    processing_tier = Column(String) # "Tier 1: OCR+Regex", "Tier 2: LLM Fallback", "Manual"
    
    # Adaptive Allocation Fields
    vulnerability_score = Column(Integer, default=0) # 0-100 based on economic triggers
    vulnerability_reasons = Column(Text) # AI reasoning for the score
    matched_schemes = Column(Text) # JSON string of recommended schemes
    
    # Fraud Detection Fields
    fraud_score = Column(Integer, default=0) # 0-100 anomaly level
    fraud_flags = Column(Text) # Flags like "Duplicate Document", "IP Anomaly"
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Scheme(Base):
    __tablename__ = "schemes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, unique=True)
    description = Column(Text)
    eligibility_criteria = Column(Text) # Keywords like "Below Poverty Line", "Farmer", "Disability"
    department = Column(String)
    funding_per_capita = Column(Integer)
    is_active = Column(Boolean, default=True)

