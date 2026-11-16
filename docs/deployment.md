# Deployment

How the web app is configured and how it runs on AWS. For running it locally, see
the [README](../README.md#running-it-locally) and [development.md](development.md).

## Web app settings

Settings are environment variables:

| Variable | Default | |
|---|---|---|
| `APP_PASSWORD` | unset | Login password (user name `APP_USERNAME`, default `etf`); unset means no login, so set it on any public server |
| `DAILY_BUDGET_USD` | `2.0` | Claude spending of all conversations together per day (UTC); then questions are refused until the next day |
| `CONCURRENCY` | `4` | Questions answered at the same time; more wait in a queue |
| `MAX_QUESTION_CHARS` | `1000` | Longest question accepted |
| `HOST`, `PORT` | `127.0.0.1`, `7860` | Where the server listens; `0.0.0.0` in a container |

`GET /health` returns 503 when search isn't working, for uptime checks.
`/documents/<file>` serves only the PDFs listed in `manifest.csv`, never the rest of `data/`.

## AWS

One EC2 instance (t4g.small, ARM) runs `deploy/docker-compose.prod.yml`: Caddy for
HTTPS, the web app, and OpenSearch without the plugins the app doesn't use. Everything
in AWS is one CloudFormation stack, `deploy/aws/stack.yaml`, named `etf-assistant`.

| In AWS | For |
|---|---|
| EC2 instance + Elastic IP | the server; ports 80 and 443 only, shell access through Session Manager |
| ECR repositories `etf-assistant/app`, `etf-assistant/opensearch` | the container images |
| S3 bucket | `data.tar.gz` (the documents) and the deploy files |
| Parameter Store `/etf-assistant/*` | the app's secrets: `APP_PASSWORD`, `ANTHROPIC_API_KEY`; optional settings such as `DAILY_BUDGET_USD` |

```bash
export AWS_PROFILE=etf
deploy/push-data.sh       # after the PDFs changed (and ingest + index locally): upload the documents
deploy/deploy.sh          # build and push the images of the current commit, update the server
```

`deploy.sh` runs `deploy/remote.sh` on the server through Systems Manager. That script
fetches the documents and secrets, pulls the images, rebuilds the search index and waits
for `/health`. To roll back, check out an earlier commit and run `deploy.sh` again.

```bash
# set or change a secret, then deploy again to apply it
aws ssm put-parameter --name /etf-assistant/ANTHROPIC_API_KEY --type SecureString --overwrite --value ...
# read the login password (user name: etf)
aws ssm get-parameter --name /etf-assistant/APP_PASSWORD --with-decryption --query Parameter.Value --output text
# a shell on the server (needs the Session Manager plugin); the app is in /opt/etf-assistant
aws ssm start-session --target $(aws cloudformation describe-stacks --stack-name etf-assistant \
    --query "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue" --output text)
```

## CI/CD

Every push to any branch runs the tests (`.github/workflows/ci.yml`). Pushes to `main`
also deploy through `.github/workflows/deploy.yml`, which runs the same `deploy.sh` as
the stack's `etf-assistant-github-deploy` user. Its access key is stored as the
repository secrets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. To replace the key:

```bash
aws iam create-access-key --user-name etf-assistant-github-deploy   # then: gh secret set ... for both values
aws iam delete-access-key --user-name etf-assistant-github-deploy --access-key-id <the old one>
```

## Removing everything

Delete that user's access key, run `aws s3 rm s3://<bucket> --recursive`, then
`aws cloudformation delete-stack --stack-name etf-assistant`, then delete the parameters
under `/etf-assistant/`.
