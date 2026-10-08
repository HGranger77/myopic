"""Defines the myopic pipeline as an Argo CronWorkflow, authored with Hera.

Run this script to (re)generate k8s/argo-workflow.yaml:

    venv-myopic/bin/python workflows/myopic_workflow.py

One Argo task per DAG stage (no per-article/per-story fan-out except
`discover`, which still fans out one task per configured source). Each stage
loads its model (if any) exactly once and loops over everything it owns,
rather than loading N times for N articles. Only `run_id` threads through
Argo parameters between tasks - everything else is a direct Postgres query
scoped by run_id, since each stage is now a single task anyway.
"""
from pathlib import Path

import yaml
from hera.workflows import (
    Container,
    ConfigMapEnvFrom,
    ConfigMapVolume,
    CronWorkflow,
    DAG,
    Parameter,
    Resources,
    RetryStrategy,
    SecretEnvFrom,
)
from hera.workflows.models import ContinueOn, PodGC, ValueFrom

IMAGE = "myopic-batch:latest"
NAMESPACE = "default"
ENV_FROM = [
    ConfigMapEnvFrom(name="myopic-db-config"),
    SecretEnvFrom(name="postgres-secret"),
]
OUT_DIR = "/tmp/outputs"

SOURCES_VOLUME = ConfigMapVolume(
    name="myopic-sources", mount_path="/app/config/sources.yaml", sub_path="sources.yaml"
)

# Source module names are structural (part of the DAG shape, like the stages
# themselves), not a runtime value - read once here, at generation time, same
# file the discover Container reads again at runtime via the ConfigMap mount
# for its own per-source front-page config.
_SOURCES_CONFIG = yaml.safe_load((Path(__file__).parent.parent / "config" / "sources.yaml").read_text())
SOURCE_MODULES = [s["module"] for s in _SOURCES_CONFIG["sources"]]


def _run_id_arg() -> dict:
    return {"run-id": "{{tasks.start-run.outputs.parameters.run-id}}"}


with CronWorkflow(
    name="myopic-batch",
    namespace=NAMESPACE,
    entrypoint="pipeline",
    schedule="0 6 * * *",
    timezone="UTC",
    concurrency_policy="Forbid",
    starting_deadline_seconds=300,
    successful_jobs_history_limit=3,
    failed_jobs_history_limit=3,
    service_account_name="default",
    # every task runs, does its work, and exits - nothing is ever left
    # standing between runs or between stages within a run.
    pod_gc=PodGC(strategy="OnPodCompletion"),
) as w:
    start_run = Container(
        name="start-run",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "start-run", "--out-dir", OUT_DIR],
        env_from=ENV_FROM,
        outputs=[Parameter(name="run-id", value_from=ValueFrom(path=f"{OUT_DIR}/run_id"))],
        resources=Resources(cpu_request="50m", cpu_limit="200m", memory_request="64Mi", memory_limit="128Mi"),
    )

    discover = Container(
        name="discover",
        image=IMAGE,
        image_pull_policy="Never",
        command=[
            "python", "run_batch.py", "discover",
            "--run-id", "{{inputs.parameters.run-id}}",
            "--source", "{{inputs.parameters.source}}",
        ],
        inputs=[Parameter(name="run-id"), Parameter(name="source")],
        env_from=ENV_FROM,
        volumes=[SOURCES_VOLUME],
        retry_strategy=RetryStrategy(limit="2"),
        resources=Resources(cpu_request="200m", cpu_limit="500m", memory_request="256Mi", memory_limit="512Mi"),
    )

    fetch_article_bodies = Container(
        name="fetch-article-bodies",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "fetch-article-bodies"],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="256Mi", memory_limit="512Mi"),
    )

    filter_article = Container(
        name="filter-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "filter-article", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="768Mi", memory_limit="1536Mi"),
    )

    embed_article = Container(
        name="embed-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "embed-article"],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="512Mi", memory_limit="1Gi"),
    )

    cluster_stories = Container(
        name="cluster-stories",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "cluster-stories", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="200m", cpu_limit="500m", memory_request="256Mi", memory_limit="512Mi"),
    )

    summarize_story = Container(
        name="summarize-story",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "summarize-story", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="500m", cpu_limit="1", memory_request="1Gi", memory_limit="2Gi"),
    )

    sentiment_article = Container(
        name="sentiment-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "sentiment-article", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="512Mi", memory_limit="1Gi"),
    )

    story_entities = Container(
        name="story-entities",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "story-entities", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="768Mi", memory_limit="1536Mi"),
    )

    finalize = Container(
        name="finalize",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "finalize", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env_from=ENV_FROM,
        resources=Resources(cpu_request="100m", cpu_limit="300m", memory_request="128Mi", memory_limit="256Mi"),
    )

    with DAG(name="pipeline"):
        sr = start_run()

        d = discover(
            name="discover",
            arguments={"run-id": "{{tasks.start-run.outputs.parameters.run-id}}", "source": "{{item}}"},
            with_items=SOURCE_MODULES,
            continue_on=ContinueOn(failed=True),
            dependencies=["start-run"],
        )

        fab = fetch_article_bodies(name="fetch-article-bodies", dependencies=["discover"])

        fa = filter_article(name="filter-article", arguments=_run_id_arg(), dependencies=["fetch-article-bodies"])

        ea = embed_article(name="embed-article", dependencies=["filter-article"])

        cs = cluster_stories(name="cluster-stories", arguments=_run_id_arg(), dependencies=["embed-article"])

        ss = summarize_story(name="summarize-story", arguments=_run_id_arg(), dependencies=["cluster-stories"])
        sa = sentiment_article(name="sentiment-article", arguments=_run_id_arg(), dependencies=["cluster-stories"])
        se = story_entities(name="story-entities", arguments=_run_id_arg(), dependencies=["cluster-stories"])

        fin = finalize(
            name="finalize",
            arguments=_run_id_arg(),
            dependencies=["summarize-story", "sentiment-article", "story-entities"],
        )


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "k8s" / "argo-workflow.yaml"
    out_path.write_text(w.to_yaml())
    print(f"wrote {out_path}")
