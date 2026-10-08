# Access

Quick links to the four myopic UIs. All of them run behind `kubectl port-forward`
on minikube — none are reachable until exposed, and none survive a terminal
restart. From the repo root:

```bash
make expose           # all three port-forwards at once (stories + API share one)
make expose-stories   # just the news viewer + watchlist API
make expose-argo      # just the Argo Workflows UI
make expose-grafana   # just Grafana
```

| UI | URL | Notes |
|---|---|---|
| News viewer | [http://localhost:8000/stories](http://localhost:8000/stories) | Matched & summarized stories, newest first |
| Watchlist API | [http://localhost:8000/docs](http://localhost:8000/docs) | FastAPI's Swagger UI — same service/port as the viewer |
| Argo Workflows | [https://localhost:2746](https://localhost:2746) | Self-signed cert — click through the browser warning |
| Grafana | [http://localhost:3000](http://localhost:3000) | Login `admin` / `admin` |
