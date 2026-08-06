# Deploying Documentation

PsychScanner's documentation is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and deployed to GitHub Pages at `https://saurabhr.github.io/psychscanner/`.

---

## Automatic deployment (recommended)

A GitHub Actions workflow is included at `.github/workflows/docs.yml`.
Every push to `main` automatically rebuilds and deploys the docs.

### One-time setup

1. Push the repository to GitHub (if not already done).
2. Go to **Settings → Pages** in your GitHub repository.
3. Under **Source**, select **Deploy from a branch** → branch `gh-pages`, folder `/ (root)`.
4. Click **Save**.

After the first workflow run completes, the site will be live at:
```
https://saurabhr.github.io/psychscanner/
```

### Workflow file

```yaml
# .github/workflows/docs.yml
name: Deploy docs
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: write
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install mkdocs
        run: pip install ".[mkdocs]"
      - name: Deploy to GitHub Pages
        run: mkdocs gh-deploy --force
```

---

## Manual deployment

To deploy from your local machine:

```bash
# Install mkdocs
pip install ".[mkdocs]"

# Build and deploy in one step
mkdocs gh-deploy --force
```

This pushes the built site to the `gh-pages` branch of your remote repository.

---

## Local preview

To preview docs locally before deploying:

```bash
mkdocs serve
```

The site is available at `http://127.0.0.1:8000/` with live reload on file changes.

---

## Building without deploying

```bash
mkdocs build
```

The static site is written to the `site/` directory. The `site/` folder is
excluded from git (listed in `.gitignore`) — only the source in `docs/` is
version controlled.

---

## Troubleshooting deployment

**Workflow fails with permission error:**
Check that the workflow has `permissions: contents: write` (already included in the
provided workflow file). In some organizations, workflows need explicit permission
from repository admins.

**Site not appearing after deploy:**
- Confirm GitHub Pages is enabled (Settings → Pages) and points to `gh-pages`.
- The first deploy may take 1–2 minutes to become live.
- Force-refresh your browser cache (`Ctrl+Shift+R`).

**`site_url` mismatch:**
The `site_url` in `mkdocs.yml` is set to `https://saurabhr.github.io/psychscanner/`.
If you fork the repository, update this value to your own GitHub Pages URL.

---

## See also

- [MkDocs Material docs](https://squidfunk.github.io/mkdocs-material/)
- [GitHub Pages documentation](https://docs.github.com/en/pages)
