from fastapi import APIRouter, HTTPException

from database import operations as ops
from schemas import AliasCreate, WatchlistEntityCreate, WatchlistEntityOut, WatchlistEntityUpdate

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.get("", response_model=list[WatchlistEntityOut])
def list_entities(active_only: bool = False):
    return ops.list_watchlist_entities(active_only=active_only)


@router.post("", response_model=WatchlistEntityOut, status_code=201)
def create_entity(payload: WatchlistEntityCreate):
    return ops.create_watchlist_entity(
        payload.canonical_name, payload.entity_type, payload.notes
    )


@router.get("/{entity_id}", response_model=WatchlistEntityOut)
def get_entity(entity_id: int):
    entity = ops.get_watchlist_entity(entity_id)
    if entity is None:
        raise HTTPException(404, "watchlist entity not found")
    return entity


@router.patch("/{entity_id}", response_model=WatchlistEntityOut)
def update_entity(entity_id: int, payload: WatchlistEntityUpdate):
    entity = ops.update_watchlist_entity(
        entity_id, **payload.model_dump(exclude_unset=True)
    )
    if entity is None:
        raise HTTPException(404, "watchlist entity not found")
    return entity


@router.delete("/{entity_id}", status_code=204)
def delete_entity(entity_id: int):
    if not ops.delete_watchlist_entity(entity_id):
        raise HTTPException(404, "watchlist entity not found")


@router.post("/{entity_id}/aliases", response_model=WatchlistEntityOut, status_code=201)
def add_alias(entity_id: int, payload: AliasCreate):
    if ops.add_alias(entity_id, payload.alias) is None:
        raise HTTPException(404, "watchlist entity not found")
    return ops.get_watchlist_entity(entity_id)


@router.delete("/aliases/{alias_id}", status_code=204)
def remove_alias(alias_id: int):
    if not ops.remove_alias(alias_id):
        raise HTTPException(404, "alias not found")
