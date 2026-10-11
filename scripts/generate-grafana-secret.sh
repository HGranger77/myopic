#!/usr/bin/env bash
# Regenerates k8s/monitoring/grafana-secret-values.yaml from .env. Kept as a script
# rather than inline in the Makefile because it's a multi-line heredoc and
# because `make cluster-monitoring` needs to re-run it idempotently.
#
# Both Grafana datasources (Prometheus + Postgres) have to live in this one
# file - Helm doesn't merge YAML lists across multiple `-f` values files, so
# splitting them across two files silently drops one. The UIDs are pinned
# (myopic-prometheus/myopic-postgres) so dashboard JSON that references them
# by UID doesn't break if Grafana's storage resets (persistence is
# deliberately disabled - see README). `database` is set both top-level
# (what the backend connects with) and again under `jsonData` (what the
# grafana-postgresql-datasource query editor's frontend checks before
# letting a query run - panels fail client-side without it even though the
# backend connects fine).
set -euo pipefail
cd "$(dirname "$0")/.."

set -a
source .env
set +a

cat > k8s/monitoring/grafana-secret-values.yaml <<EOF
datasources:
  datasources.yaml:
    apiVersion: 1
    datasources:
      - name: Prometheus
        uid: myopic-prometheus
        type: prometheus
        access: proxy
        url: http://prometheus-server.monitoring.svc.cluster.local
        isDefault: true
      - name: Postgres
        uid: myopic-postgres
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
          database: $POSTGRES_DB
EOF

echo "wrote k8s/monitoring/grafana-secret-values.yaml"
