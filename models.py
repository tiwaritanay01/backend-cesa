from sqlalchemy import Column, Integer, String, Text, DateTime
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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
