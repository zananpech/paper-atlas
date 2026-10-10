from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langfuse import Langfuse

logger = logging.getLogger(__name__)

_DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env.local"
_langfuse_instance: Optional[Langfuse] = None


def init_langfuse(
    env_file: Optional[Path | str] = None,
    verify_auth: bool = True,
    force_new: bool = False,
) -> Langfuse:
    """Initialize and configure a Langfuse client from an environment file.

    Args:
        env_file: Path to the .env file containing Langfuse credentials.
            Defaults to ``src/paper_atlas/.env.local``.
        verify_auth: Whether to perform an immediate authentication check against Langfuse.
        force_new: If True, recreate the cached Langfuse instance.

    Returns:
        Configured Langfuse client instance.

    Raises:
        ValueError: If required Langfuse API keys are missing.
        RuntimeError: If verify_auth is True and the auth check fails.
    """
    global _langfuse_instance

    if _langfuse_instance is not None and not force_new and env_file is None:
        return _langfuse_instance

    env_path = Path(env_file) if env_file is not None else _DEFAULT_ENV_PATH

    if env_path.is_file():
        logger.info("Loading Langfuse configuration from %s", env_path)
        load_dotenv(dotenv_path=env_path, override=True)
    elif env_file is not None:
        raise FileNotFoundError(f"Specified environment file does not exist: {env_path}")

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

    if not public_key or not secret_key:
        raise ValueError(
            "Langfuse initialization failed: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            f"must be defined (checked {env_path})."
        )

    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
    )

    if verify_auth:
        if not client.auth_check():
            raise RuntimeError(
                f"Langfuse authentication check failed for host: {base_url or 'default cloud'}"
            )
        logger.info("Langfuse authentication check succeeded (base_url=%s)", base_url)

    _langfuse_instance = client
    return client


def get_langfuse() -> Langfuse:
    """Retrieve the active Langfuse client singleton, initializing if needed."""
    global _langfuse_instance
    if _langfuse_instance is None:
        _langfuse_instance = init_langfuse(verify_auth=False)
    return _langfuse_instance


def main() -> None:
    """Verify Langfuse initialization from .env.local via CLI."""

    target_env = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_ENV_PATH
    logger.info("Initializing Langfuse using: %s", target_env)

    try:
        client = init_langfuse(env_file=target_env, verify_auth=True, force_new=True)
        logger.info("Successfully initialized and authenticated Langfuse!")
        logger.info("Base URL: %s", getattr(client, "_base_url", "N/A"))
        logger.info("Public Key: %s...", (os.getenv("LANGFUSE_PUBLIC_KEY") or "")[:10])
    except Exception:
        logger.exception("Failed to initialize Langfuse")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()

