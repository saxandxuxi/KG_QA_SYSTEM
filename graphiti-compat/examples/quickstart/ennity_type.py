from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# 自定义实体类型
class Person(BaseModel):
    """具有政治背景信息的人物实体"""

    current_position: Optional[str] = Field(None, description="当前职位")
    previous_positions: Optional[list[str]] = Field(None, description="过往职位列表")
    serving_state: Optional[str] = Field(None, description="任职州")

class GovernmentPosition(BaseModel):
    """政府职位实体"""
    title: Optional[str] = Field(None, description="职位名称")
    jurisdiction: Optional[str] = Field(None, description="管辖范围")
    department: Optional[str] = Field(None, description="所属部门")
    level: Optional[str] = Field(None, description="职位级别（州级/市级等）")

class PoliticalEntity(BaseModel):
    """政治实体（如州、城市）"""

    type: Optional[str] = Field(None, description="实体类型（州/城市等）")
    country: Optional[str] = Field(None, description="所属国家")

# 自定义关系类型
class HoldsPosition(BaseModel):
    """人物与职位之间的任职关系"""
    # start_date: Optional[datetime] = Field(None, description="任职开始日期")
    # end_date: Optional[datetime] = Field(None, description="任职结束日期")
    is_current: Optional[bool] = Field(None, description="是否为现任")
    responsibilities: Optional[str] = Field(None, description="职责描述")

class PreviouslyHeld(BaseModel):
    """人物与过往职位之间的关系"""
    # start_date: Optional[datetime] = Field(None, description="开始日期")
    # end_date: Optional[datetime] = Field(None, description="结束日期")
    achievements: Optional[str] = Field(None, description="任职期间成就")

class Governs(BaseModel):
    """官员与管辖区域之间的治理关系"""
    jurisdiction_level: Optional[str] = Field(None, description="管辖级别")
    term_length: Optional[str] = Field(None, description="任期长度")
    key_initiatives: Optional[list[str]] = Field(None, description="主要举措")