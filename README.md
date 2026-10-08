# myopic

Daily batch pipeline that scrapes nine.com.au, runs local NER on each article, and generates a
summary only for stories that mention an entity on a user-managed watchlist of "reportable
entities." The watchlist is managed through a small FastAPI CRUD service; everything runs
locally (spaCy + a local transformers summarization model — no external LLM APIs).

## Local development

```bash
python3 -m venv venv-myopic
venv-myopic/bin/pip install -r requirements.txt
venv-myopic/bin/python -m spacy download en_core_web_sm

cp .env.example .env   # adjust credentials if needed
docker compose up -d postgres

venv-myopic/bin/python run_batch.py                 # run the pipeline once
venv-myopic/bin/uvicorn main:app --reload            # watchlist API on :8000, story viewer at :8000/stories
venv-myopic/bin/python -m pytest                     # test suite (needs postgres up)
```

Postgres is exposed on host port **5433** (not 5432) to avoid clashing with other local
Postgres containers — see `docker-compose.yml`.

## Docker

```bash
docker compose build
docker compose up -d postgres api
docker compose run --rm batch
```

## Kubernetes (minikube)

```bash
minikube start --driver=docker
minikube image build -t myopic-api:latest -f Dockerfile.api .
minikube image build -t myopic-batch:latest -f Dockerfile.batch .

set -a; source .env; set +a
kubectl create secret generic postgres-secret \
  --from-literal=POSTGRES_USER="$POSTGRES_USER" \
  --from-literal=POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  --from-literal=POSTGRES_DB="$POSTGRES_DB" \
  --dry-run=client -o yaml > k8s/secret.yaml
kubectl create configmap myopic-sources --from-file=config/sources.yaml --dry-run=client -o yaml > /tmp/myopic-sources.yaml
kubectl create configmap myopic-schema --from-file=database/schema.sql --dry-run=client -o yaml > /tmp/myopic-schema.yaml

kubectl apply -f k8s/configmap.yaml -f k8s/secret.yaml -f /tmp/myopic-sources.yaml -f /tmp/myopic-schema.yaml
kubectl apply -f k8s/postgres-pvc.yaml -f k8s/postgres-deployment.yaml -f k8s/postgres-service.yaml
kubectl wait --for=condition=available deployment/postgres --timeout=120s
kubectl apply -f k8s/watchlist-api-deployment.yaml -f k8s/watchlist-api-service.yaml
```

`k8s/secret.yaml` is generated from `.env` and is gitignored — only `k8s/secret.example.yaml`
is committed. `myopic-sources` and `myopic-schema` ConfigMaps are generated from their
canonical source files (`config/sources.yaml`, `database/schema.sql`) rather than duplicated
in a committed manifest, so they can't drift out of sync.

### Batch orchestration: Argo Workflows

The daily batch job runs as an Argo CronWorkflow (`discover -> process-article [fanned out,
parallelism 8] -> summarize -> finalize`), authored in Python via Hera
(`workflows/myopic_workflow.py` generates `k8s/argo-workflow.yaml`).

```bash
kubectl create namespace argo
kubectl apply --server-side -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v4.1.4/install.yaml
kubectl patch deployment argo-server -n argo --type='json' \
  -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args", "value": ["server", "--auth-mode=server"]}]'

# the workflow runs in `default` alongside postgres/the API, so that namespace's
# default ServiceAccount needs permission to report step outputs back to Argo:
kubectl apply -f k8s/argo-rbac.yaml

venv-myopic/bin/python workflows/myopic_workflow.py   # regenerate k8s/argo-workflow.yaml after editing it
kubectl apply -f k8s/argo-workflow.yaml

# trigger a one-off run to test it:
argo submit --from cronworkflow/myopic-batch -n default
argo get @latest -n default
```

See `workflows/myopic_workflow.py` for why `continueOn` and `depends` can't be mixed in the
same DAG, and why the fan-out needs an explicit `parallelism` cap.

### Monitoring: Prometheus + Grafana

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

kubectl create namespace monitoring
helm install prometheus prometheus-community/prometheus -n monitoring -f k8s/monitoring-values-prometheus.yaml

# grafana-secret-values.yaml holds the Postgres datasource password - generate it from .env
# (gitignored; see k8s/grafana-secret-values.example.yaml for the template):
set -a; source .env; set +a
cat > k8s/grafana-secret-values.yaml <<EOF
datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus-server.monitoring.svc.cluster.local
        isDefault: true
      - name: Postgres
        type: postgres
        access: proxy
        url: postgres.default.svc.cluster.local:5432
        database: $POSTGRES_DB
        user: $POSTGRES_USER
        secureJsonData:
          password: $POSTGRES_PASSWORD
        jsonData:
          sslmode: disable
          postgresVersion: 1600
EOF

kubectl apply -f k8s/grafana-dashboard-configmap.yaml
helm install grafana grafana/grafana -n monitoring \
  -f k8s/monitoring-values-grafana.yaml \
  -f k8s/grafana-secret-values.yaml

kubectl port-forward -n monitoring svc/grafana 3000:80
# open http://localhost:3000, log in as admin/admin, open the "myopic pipeline" dashboard
```

The dashboard (`k8s/grafana-dashboard-configmap.yaml`) has 4 panels: batch run duration and
articles-matched/summarized counts (queried straight from the `scrape_runs` table via a
Postgres datasource — no extra instrumentation needed), plus pod CPU and memory usage
(from Prometheus, scraped via cAdvisor/kube-state-metrics/node-exporter). It's auto-loaded by
Grafana's dashboard sidecar from any ConfigMap labeled `grafana_dashboard: "1"` in the
`monitoring` namespace. `persistence.enabled: false` for Grafana is intentional — everything
is provisioned as code, so there's nothing to lose on pod restart.
