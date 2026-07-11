# Kubernetes manifests — v2 broker control plane (starting point, **not deployed**)

> ⚠️ **These manifests are provided as code. Nothing here is applied by CI or by any
> automated agent.** They contain `REPLACE_ME` placeholders (image, DSN, token). Review
> them, set the placeholders, and `kubectl apply` **yourself**. Applying to a cluster —
> and standing up the LLM servers — is a deliberate operator action.

## What's here

| File | What it stands up |
|---|---|
| `config.yaml` | `broker-parrot` Namespace + `broker-config` ConfigMap + `broker-secrets` Secret |
| `broker-web.yaml` | `queue-broker-web` Deployment + Service (the control-plane HTTP face + panel) |
| `broker-worker.yaml` | `queue-broker-worker` Deployments (one per project/resource lane) |
| `llm-servers.yaml` | broker-managed `ollama` + `vllm` Deployments + Services |

Every manifest runs the **real console scripts** shipped by `queue_workflows`
(`queue-broker-web`, `queue-broker-worker`) and reads the **real config knobs**
(`QUEUE_WORKFLOWS_DB_BACKEND=pg`, `AI_LEADS_DB_URL`, `AI_LEADS_OLLAMA_URL` /
`AI_LEADS_VLLM_URL`, `QUEUE_WORKFLOWS_BROKER_WEB_TOKEN`).

## Before applying

1. Build an image bundling `queue_workflows` (+ each project's worker handlers) and set the
   `REPLACE_ME/...:latest` image refs.
2. Put the broker Postgres DSN + the web bearer token into `broker-secrets` (never commit
   real secret values).
3. Set GPU `nodeSelector`/`tolerations` for your cluster on the GPU workloads.
4. Front `broker-web` with an Ingress if it must be reachable outside the cluster (and keep
   the bearer token set).

## Apply order (once you've reviewed + edited)

```bash
kubectl apply -f config.yaml
kubectl apply -f llm-servers.yaml   # bring the LLM Services up first
kubectl apply -f broker-web.yaml
kubectl apply -f broker-worker.yaml
```

## Validation without a cluster

Client-side **schema** validation (needs a cluster or `kubeconform`/`kubeval`) is not run
here. Only YAML **syntax** is checked in this repo. To schema-check locally:

```bash
kubeconform -strict -summary deploy/k8s/*.yaml
```

## What is still design-only

The **broker controller** that reconciles LLM capacity from live queue demand — scaling /
(re)modeling the `vllm` Deployment instead of running it statically — is **not built**. See
[`docs/broker_k8s_llm.md`](../../docs/broker_k8s_llm.md).
