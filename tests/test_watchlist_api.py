from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_crud_cycle():
    create_resp = client.post("/watchlist", json={"canonical_name": "Joe Biden"})
    assert create_resp.status_code == 201
    entity = create_resp.json()
    entity_id = entity["id"]
    assert [a["alias"] for a in entity["aliases"]] == ["Joe Biden"]

    list_resp = client.get("/watchlist")
    assert list_resp.status_code == 200
    assert any(e["id"] == entity_id for e in list_resp.json())

    alias_resp = client.post(f"/watchlist/{entity_id}/aliases", json={"alias": "Biden"})
    assert alias_resp.status_code == 201
    assert sorted(a["alias"] for a in alias_resp.json()["aliases"]) == ["Biden", "Joe Biden"]

    patch_resp = client.patch(f"/watchlist/{entity_id}", json={"is_active": False})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["is_active"] is False

    delete_resp = client.delete(f"/watchlist/{entity_id}")
    assert delete_resp.status_code == 204

    get_resp = client.get(f"/watchlist/{entity_id}")
    assert get_resp.status_code == 404


def test_get_missing_entity_404():
    assert client.get("/watchlist/999999").status_code == 404


def test_add_alias_to_missing_entity_404():
    resp = client.post("/watchlist/999999/aliases", json={"alias": "x"})
    assert resp.status_code == 404
