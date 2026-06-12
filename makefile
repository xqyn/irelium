# ── Sphinx ────────────────────────────────────────────────
docs-build:
	sphinx-build -b html docs _build/html

docs-serve:
	python -m http.server 8080 --directory _build/html

docs-clean:
	rm -rf _build/ site/

docs-deploy:
	sphinx-build -b html docs _build/html
	ghp-import -n -p -f -r irelium _build/html

# ── MkDocs ────────────────────────────────────────────────
mkdocs:
	mkdocs gh-deploy --force --remote-name irelium

mkdocs-build:
	mkdocs build

mkdocs-serve:
	mkdocs serve

# ── Both ──────────────────────────────────────────────────
docs-all: docs-build mkdocs-build