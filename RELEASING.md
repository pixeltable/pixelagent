# Releasing pixelagent

```bash
# 1. bump the version
#    pyproject.toml  ->  version = "X.Y.Z"
uv lock            # keep uv.lock in step; CI runs `uv sync --locked` and will fail otherwise

# 2. land it on main through a PR, then
git tag vX.Y.Z && git push origin vX.Y.Z
```

The tag push runs `.github/workflows/release.yml`, which builds the sdist and
wheel, checks the tag matches `pyproject.toml`, installs the built wheel into a
clean environment and exercises it, then publishes to PyPI.

## There is no PyPI token

Publishing uses PyPI Trusted Publishing. PyPI mints a short-lived credential
from the workflow's OIDC identity, so there is no long-lived secret to leak,
rotate, or paste into a shell.

One-time setup, on PyPI under **Manage project -> Publishing -> Add a pending publisher**:

| field | value |
|---|---|
| owner | `pixeltable` |
| repository | `pixelagent` |
| workflow | `release.yml` |
| environment | `pypi` |

To require a human approval before each publish, add a required reviewer to the
`pypi` environment under **Settings -> Environments**.

## Dry run

Run the workflow manually from the Actions tab with `dry_run` left checked. It
builds and verifies the artifact without publishing anything.

## Why the release script is gone

`scripts/release.sh` read a long-lived `PYPI_API_KEY` out of
`~/.pixeltable/config.toml`, required a git remote named `home` that a normal
clone does not have, and called `poetry build` after the project had moved to
uv. It also published without ever installing what it built, which is how
0.1.x shipped a package that could not import against a current Pixeltable.
