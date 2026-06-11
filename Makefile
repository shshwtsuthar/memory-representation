.PHONY: validate-docs release-notes

validate-docs:
	python scripts/release/validate_repo_docs.py

release-notes:
	cat docs/releases/v0.1-workshop-submission.md
