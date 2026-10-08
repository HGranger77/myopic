VENV := venv-myopic/bin

.PHONY: install up down build batch check expose expose-stories expose-argo expose-grafana

install:
	$(VENV)/pip install -r requirements.txt
	$(VENV)/python -m spacy download en_core_web_sm

up:
	docker compose up -d postgres api

down:
	docker compose down

build:
	docker compose build

batch:
	docker compose run --rm batch

check:
	$(VENV)/python -m pytest

# --- k8s access (requires the minikube stack to be deployed and running) ---

expose-stories:
	@echo "Stories:  http://localhost:8000/stories"
	kubectl port-forward svc/watchlist-api 8000:8000

expose-argo:
	@echo "Argo UI:  https://localhost:2746  (self-signed cert - click through the warning)"
	kubectl port-forward -n argo svc/argo-server 2746:2746

expose-grafana:
	@echo "Grafana:  http://localhost:3000  (login admin/admin)"
	kubectl port-forward -n monitoring svc/grafana 3000:80

# All three at once; Ctrl+C stops all three together.
expose:
	@echo "Stories:  http://localhost:8000/stories"
	@echo "Argo UI:  https://localhost:2746  (self-signed cert - click through the warning)"
	@echo "Grafana:  http://localhost:3000  (login admin/admin)"
	@trap 'kill 0' INT TERM EXIT; \
	kubectl port-forward svc/watchlist-api 8000:8000 & \
	kubectl port-forward -n argo svc/argo-server 2746:2746 & \
	kubectl port-forward -n monitoring svc/grafana 3000:80 & \
	wait
