# Development

Local development, Docker, Kubernetes deployment, and monitoring setup. See
[README.md](README.md) for what the project does and how it's architected.

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

The whole stack - minikube itself, both images, Postgres, the watchlist API, Argo Workflows,
the CronWorkflow, and Prometheus/Grafana - comes up with one command:

```bash
make cluster-deploy
```

This is also how to **recover the environment** after minikube/Docker gets stopped (e.g. a
host/WSL restart) or the VM is lost: every step is either `kubectl apply` or `helm upgrade
--install`, so re-running it on a partially-up cluster is safe and just reconciles whatever's
missing. `minikube stop` (exposed as `make cluster-down`) preserves the VM's disk - PVC data,
installed Helm releases, images already built into minikube's store - so most of the time
`make cluster-deploy` just needs to restart minikube and re-apply manifests against pods that
come back on their own; it does NOT need `minikube delete` first. Use `minikube delete &&
minikube start` by hand only for a genuine from-scratch wipe.

Each stage can also be run on its own (useful when only one part needs redoing, e.g. after
changing just `config/sources.yaml` or just the Grafana dashboard):

| Target | What it does |
|---|---|
| `make cluster-up` | `minikube start --driver=docker` |
| `make cluster-images` | Builds `myopic-api`/`myopic-batch` straight into minikube's image store |
| `make cluster-secrets` | Regenerates `postgres-secret` + the `myopic-sources`/`myopic-schema` ConfigMaps from `.env`/`config/sources.yaml`/`database/schema.sql`, applies `k8s/shared/configmap.yaml` |
| `make cluster-postgres` | Applies the Postgres PVC/Deployment/Service, waits for it to be available |
| `make cluster-api` | Applies the watchlist API Deployment/Service |
| `make cluster-argo` | Installs/upgrades Argo Workflows itself (CRDs, controller, server) + the RBAC binding the batch CronWorkflow needs |
| `make cluster-workflow` | Regenerates `k8s/argo/argo-workflow.yaml` from `workflows/myopic_workflow.py` and applies it |
| `make cluster-monitoring` | Installs/upgrades Prometheus + Grafana via Helm, regenerates the Grafana datasource secret |
| `make cluster-status` | `minikube status` + `kubectl get pods -A`, for a quick health check |

`k8s/shared/secret.yaml` and `k8s/monitoring/grafana-secret-values.yaml` are both generated from `.env` and
gitignored — only the `*.example.yaml` templates are committed. `myopic-sources` and
`myopic-schema` ConfigMaps are generated from their canonical source files
(`config/sources.yaml`, `database/schema.sql`) rather than duplicated in a committed manifest,
so they can't drift out of sync.

### Batch orchestration: Argo Workflows

The daily batch job runs as an Argo CronWorkflow with one task per pipeline stage - each task
loads its model (if any) once and loops over everything it owns, rather than fanning out one
pod per article. See [README.md](README.md#architecture) for the DAG diagram.

Relevance filtering runs **after** clustering and summarizing, not before: every fetched
article gets embedded and clustered into a story regardless of watchlist relevance, and every
touched story gets a summary. Only then does `tag-relevance` classify that short summary
against the watchlist and stamp the verdict onto the story's member articles -
`sentiment-article` and `article-entities` are both scoped back down to just the relevant
subset, since there's no reason to pay for either on stories nobody's watching for. This order is
deliberate, not incidental - see `tag_relevance()`'s docstring in `run_batch.py` for the
(measured, not assumed) reasons: checking the raw article text directly, before this reordering,
was both slower and less accurate than checking the much shorter story summary.

`article-entities` is a one-shot NER pass per article (never redone - an article's own text
doesn't change once fetched), and powers the per-source people/organizations/locations breakdown
on the `/stories` page, filtered down to what's actually significant to that article rather than
every incidental mention - see `nlp/matching.py`'s `filter_significant_entities()`. A
story-level equivalent (NER over every source's combined text) existed earlier but was removed -
it was never read by anything.

`translate-article` sits right after `fetch-article-bodies`: translation is autoregressive
(the same cost class as the disconnected LLM path, not `tag-relevance`'s one-shot classifier),
so it's bounded to a prefix of each article rather than the full body - see `nlp/translate.py`.
It costs nothing when every configured source is English (the common case) - the model import
itself is skipped, not just the model load, if no source's `language` is set to anything but
`"en"` in `config/sources.yaml`.

Authored in Python via Hera (`workflows/myopic_workflow.py` generates `k8s/argo/argo-workflow.yaml`
- `make cluster-workflow` does this regeneration-and-apply for you). See that file for why
`continueOn` and the modern `depends` dependency syntax can't be mixed in the same DAG.

```bash
# trigger a one-off run to test it, after `make cluster-workflow`:
argo submit --from cronworkflow/myopic-batch -n default
argo get @latest -n default
```

### Monitoring: Prometheus + Grafana

`make cluster-monitoring` installs both via Helm and regenerates
`k8s/monitoring/grafana-secret-values.yaml` (see `scripts/generate-grafana-secret.sh`). Both Grafana
datasources (Prometheus + Postgres) have to live in that one file - Helm doesn't merge YAML
lists across multiple `-f` values files - and their UIDs are pinned
(`myopic-prometheus`/`myopic-postgres`) so the dashboard JSON that references them by UID
doesn't break when Grafana's storage resets (`persistence.enabled: false` is deliberate -
everything is provisioned as code, so there's nothing to lose on pod restart).

```bash
make expose-grafana
# open http://localhost:3000, log in as admin/admin, open the "myopic pipeline" dashboard
```

The dashboard (`k8s/monitoring/grafana-dashboard-configmap.yaml`) has 4 panels: batch run duration and
articles-matched/summarized counts (queried straight from the `scrape_runs` table via a
Postgres datasource — no extra instrumentation needed), plus pod CPU and memory usage
(from Prometheus, scraped via cAdvisor/kube-state-metrics/node-exporter). It's auto-loaded by
Grafana's dashboard sidecar from any ConfigMap labeled `grafana_dashboard: "1"` in the
`monitoring` namespace.

### Accessing everything

```bash
make expose            # stories page, Argo UI, and Grafana all at once (Ctrl+C stops all three)
make expose-stories    # just http://localhost:8000/stories
make expose-argo       # just https://localhost:2746
make expose-grafana    # just http://localhost:3000
```
