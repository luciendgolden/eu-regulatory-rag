"""Allow `python -m ingestion.pipeline` to invoke the CLI."""

from ingestion.pipeline import main
import sys

sys.exit(main())
