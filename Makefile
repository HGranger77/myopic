VENV := venv-myopic/bin
ARGO_VERSION := v4.1.4

.PHONY: install up down build batch check expose expose-stories expose-argo expose-grafana \
	cluster-up cluster-images cluster-secrets cluster-postgres cluster-api cluster-argo \
	cluster-workflow cluster-monitoring cluster-deploy cluster-status cluster-down

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

# --- full minikube environment rebuild ---
#
# `make cluster-deploy` runs every step below in order and is safe to re-run
# on a half-up cluster (every step is either kubectl apply or helm upgrade
# --install) - this is how to recover after minikube/docker gets stopped or
# the VM is lost, not just how to set it up the first time. See the
# "Kubernetes (minikube)" section of README.md for what each step does and
# why, and `scripts/generate-grafana-secret.sh` for why the Grafana
# datasource file can't just be a static committed YAML.

cluster-up:
	minikube start --driver=docker

cluster-images:
	minikube image build -t myopic-api:latest -f Dockerfile.api .
	minikube image build -t myopic-batch:latest -f Dockerfile.batch .

cluster-secrets:
	set -a; . ./.env; set +a; \
	kubectl create secret generic postgres-secret \
	  --from-literal=POSTGRES_USER="$$POSTGRES_USER" \
	  --from-literal=POSTGRES_PASSWORD="$$POSTGRES_PASSWORD" \
	  --from-literal=POSTGRES_DB="$$POSTGRES_DB" \
	  --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap myopic-sources --from-file=config/sources.yaml --dry-run=client -o yaml | kubectl apply -f -
	kubectl create configmap myopic-schema --from-file=database/schema.sql --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/shared/configmap.yaml

cluster-postgres: cluster-secrets
	kubectl apply -f k8s/postgres/postgres-pvc.yaml -f k8s/postgres/postgres-deployment.yaml -f k8s/postgres/postgres-service.yaml
	kubectl wait --for=condition=available deployment/postgres --timeout=180s

cluster-api: cluster-images cluster-postgres
	kubectl apply -f k8s/watchlist-api/watchlist-api-deployment.yaml -f k8s/watchlist-api/watchlist-api-service.yaml
	kubectl wait --for=condition=available deployment/watchlist-api --timeout=120s

cluster-argo:
	kubectl get namespace argo >/dev/null 2>&1 || kubectl create namespace argo
	# --force-conflicts: re-applying install.yaml on a cluster that's already
	# been through the `kubectl patch` below otherwise fails server-side
	# apply with a field-manager conflict on argo-server's container args.
	kubectl apply --server-side --force-conflicts -n argo -f https://github.com/argoproj/argo-workflows/releases/download/$(ARGO_VERSION)/install.yaml
	kubectl patch deployment argo-server -n argo --type='json' \
	  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args", "value": ["server", "--auth-mode=server"]}]' || true
	kubectl apply -f k8s/argo/argo-rbac.yaml
	kubectl wait --for=condition=available deployment/argo-server -n argo --timeout=180s
	kubectl wait --for=condition=available deployment/workflow-controller -n argo --timeout=180s

cluster-workflow: cluster-argo cluster-images
	$(VENV)/python workflows/myopic_workflow.py
	kubectl apply -f k8s/argo/argo-workflow.yaml

cluster-monitoring:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	kubectl get namespace monitoring >/dev/null 2>&1 || kubectl create namespace monitoring
	helm upgrade --install prometheus prometheus-community/prometheus -n monitoring -f k8s/monitoring/monitoring-values-prometheus.yaml
	./scripts/generate-grafana-secret.sh
	kubectl apply -f k8s/monitoring/grafana-dashboard-configmap.yaml
	helm upgrade --install grafana grafana/grafana -n monitoring \
	  -f k8s/monitoring/monitoring-values-grafana.yaml \
	  -f k8s/monitoring/grafana-secret-values.yaml

# The whole stack, in dependency order: minikube -> images -> postgres ->
# watchlist API -> Argo -> the CronWorkflow itself -> Prometheus/Grafana.
cluster-deploy: cluster-up cluster-images cluster-postgres cluster-api cluster-argo cluster-workflow cluster-monitoring
	@echo "Cluster rebuilt. Run 'make expose' to reach the stories page, Argo UI, and Grafana."

cluster-status:
	@minikube status
	@kubectl get pods -A

# Stops the minikube VM without deleting it - PVC/PV data and installed
# charts survive; `make cluster-deploy` brings everything back without a
# full reinstall. Use `minikube delete` by hand for a true from-scratch wipe.
cluster-down:
	minikube stop

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
