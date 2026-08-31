"""Validate local credentials without printing secret values."""

from __future__ import annotations

from data_pipeline.config import get_config
from data_pipeline.credentials import CredentialError, initialize_earth_engine, validate_nass_key


def main() -> None:
    cfg = get_config()
    print("Credential configuration")
    print("========================")
    print(f"NASS_API_KEY: {'set' if cfg.nass_api_key else 'missing'}")
    print(f"GEE_PROJECT: {'set' if cfg.gee_project else 'missing/optional'}")
    print(f"Data dir: {cfg.data_dir}")
    print(f"Years: {cfg.start_year}-{cfg.end_year}")

    print("\nChecking NASS Quick Stats...")
    try:
        validate_nass_key(cfg)
        print("  OK: NASS API key works.")
    except Exception as exc:
        print(f"  FAIL: {exc}")

    print("\nChecking Google Earth Engine...")
    try:
        initialize_earth_engine(cfg)
        print("  OK: Earth Engine initialized.")
    except CredentialError as exc:
        print(f"  SKIP/FAIL: {exc}")
    except Exception as exc:
        print(f"  FAIL: {exc}")


if __name__ == "__main__":
    main()
