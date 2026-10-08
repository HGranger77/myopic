from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AliasOut(BaseModel):
    id: int
    alias: str
    normalized_alias: str
    created_at: datetime


class WatchlistEntityCreate(BaseModel):
    canonical_name: str = Field(min_length=1)
    entity_type: Optional[str] = None
    notes: Optional[str] = None


class WatchlistEntityUpdate(BaseModel):
    canonical_name: Optional[str] = None
    entity_type: Optional[str] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None


class WatchlistEntityOut(BaseModel):
    id: int
    canonical_name: str
    entity_type: Optional[str]
    notes: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    aliases: list[AliasOut]


class AliasCreate(BaseModel):
    alias: str = Field(min_length=1)
