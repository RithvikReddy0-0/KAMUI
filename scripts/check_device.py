"""Check that KAMUI really runs on a device.

Usage:
    python scripts/check_device.py
    python scripts/check_device.py --device cuda
    kamui-check-device --device cuda

All logic lives in ``kamui.scripts.check_device`` so it is importable and testable.
"""

from kamui.scripts.check_device import main

if __name__ == "__main__":
    main()
