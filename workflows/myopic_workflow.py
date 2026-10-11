"""Defines the myopic pipeline as an Argo CronWorkflow, authored with Hera.

Run this script to (re)generate k8s/argo/argo-workflow.yaml:

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
    Env,
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
# Without this, stdout is fully (not line-)buffered in a non-TTY container,
# so the per-item progress prints in run_batch.py wouldn't show up in the
# Argo UI's live log view until the process exits and flushes everything at
# once - defeating their whole purpose of watching a run progress.
ENV = [Env(name="PYTHONUNBUFFERED", value="1")]
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
        env=ENV,
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
        env=ENV,
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
        env=ENV,
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="256Mi", memory_limit="512Mi"),
    )

    translate_article = Container(
        name="translate-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "translate-article"],
        env=ENV,
        env_from=ENV_FROM,
        # Needs config/sources.yaml mounted (same as discover) to know which
        # sources are non-English - nlp/translate.py's model only loads at
        # all if that check finds one. cpu_limit=4 must match
        # nlp/translate.py's _N_THREADS; runs alone in the DAG like
        # summarize-story/tag-relevance, same reasoning for taking most of
        # the node's cores. memory_limit=3Gi matches summarize-story, not
        # the smaller sentiment-article/article-entities budgets - OOMKilled
        # at 1Gi on the first live run (same generative-model size class as
        # distilbart, underestimated the same way before actually measuring).
        volumes=[SOURCES_VOLUME],
        resources=Resources(cpu_request="2", cpu_limit="4", memory_request="2Gi", memory_limit="3Gi"),
    )

    embed_article = Container(
        name="embed-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "embed-article"],
        env=ENV,
        env_from=ENV_FROM,
        resources=Resources(cpu_request="300m", cpu_limit="1", memory_request="512Mi", memory_limit="1Gi"),
    )

    cluster_stories = Container(
        name="cluster-stories",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "cluster-stories", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
        env_from=ENV_FROM,
        resources=Resources(cpu_request="200m", cpu_limit="500m", memory_request="256Mi", memory_limit="512Mi"),
    )

    summarize_story = Container(
        name="summarize-story",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "summarize-story", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
        env_from=ENV_FROM,
        # cpu_limit=4 must match nlp/summarize.py's _N_THREADS - this task
        # runs alone in the DAG (nothing else model-bearing runs
        # concurrently with it), so it can safely take most of the node's
        # cores. Bumped from 1 after the DAG reorder made this run on every
        # clustered story instead of just the relevant few - at 1 core,
        # distilbart still spawned PyTorch's default thread count (19,
        # unrelated to this cpu_limit) and the resulting oversubscription
        # made the full-volume run project to ~3.7 hours.
        resources=Resources(cpu_request="2", cpu_limit="4", memory_request="1Gi", memory_limit="2Gi"),
    )

    tag_relevance = Container(
        name="tag-relevance",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "tag-relevance", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
        env_from=ENV_FROM,
        # Currently runs nlp/tagging.py's zero-shot classifier (peak RSS
        # ~1.7GB measured directly) against each touched story's SHORT
        # SUMMARY, not the raw article text - see tag_relevance()'s
        # docstring in run_batch.py for why that distinction matters, both
        # for speed and for accuracy. nlp/relevance.py's LLM is kept wired
        # in but disconnected and would fit the same budget if swapped back
        # in (~1.9GB, via a quantized GGUF - its docstring explains why the
        # unquantized fp32 checkpoint, ~6GB, was rejected). cpu_limit=4 must
        # match whichever module's own _N_THREADS is active - this task
        # runs alone in the DAG (nothing else model-bearing runs
        # concurrently with it), so it can safely take most of the node's
        # cores for its duration without starving anything else.
        resources=Resources(cpu_request="2", cpu_limit="4", memory_request="2Gi", memory_limit="3Gi"),
    )

    sentiment_article = Container(
        name="sentiment-article",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "sentiment-article", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
        env_from=ENV_FROM,
        # cpu_limit=2 must match nlp/sentiment.py's _N_THREADS. Lower than
        # summarize-story/tag-relevance's 4 since this runs CONCURRENTLY
        # with article-entities (both depend on tag-relevance), not alone.
        resources=Resources(cpu_request="1", cpu_limit="2", memory_request="512Mi", memory_limit="1Gi"),
    )

    article_entities = Container(
        name="article-entities",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "article-entities", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
        env_from=ENV_FROM,
        # spaCy NER over one article's own body text. Runs CONCURRENTLY with
        # sentiment-article (both depend on tag-relevance), not alone.
        resources=Resources(cpu_request="200m", cpu_limit="500m", memory_request="512Mi", memory_limit="768Mi"),
    )

    finalize = Container(
        name="finalize",
        image=IMAGE,
        image_pull_policy="Never",
        command=["python", "run_batch.py", "finalize", "--run-id", "{{inputs.parameters.run-id}}"],
        inputs=[Parameter(name="run-id")],
        env=ENV,
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

        ta = translate_article(name="translate-article", dependencies=["fetch-article-bodies"])

        ea = embed_article(name="embed-article", dependencies=["translate-article"])

        cs = cluster_stories(name="cluster-stories", arguments=_run_id_arg(), dependencies=["embed-article"])

        ss = summarize_story(name="summarize-story", arguments=_run_id_arg(), dependencies=["cluster-stories"])

        tr = tag_relevance(name="tag-relevance", arguments=_run_id_arg(), dependencies=["summarize-story"])

        sa = sentiment_article(name="sentiment-article", arguments=_run_id_arg(), dependencies=["tag-relevance"])
        ae = article_entities(name="article-entities", arguments=_run_id_arg(), dependencies=["tag-relevance"])

        fin = finalize(
            name="finalize",
            arguments=_run_id_arg(),
            dependencies=["sentiment-article", "article-entities"],
        )


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "k8s" / "argo" / "argo-workflow.yaml"
    out_path.write_text(w.to_yaml())
    print(f"wrote {out_path}")
