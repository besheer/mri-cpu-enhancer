from __future__ import annotations

import json
import os
from pathlib import Path

from pynetdicom import AE, evt
from pynetdicom.sop_class import MRImageStorage, Verification

OUT = Path(os.getenv("E2E_OUT", "/e2e"))
OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob("*.dcm"):
    old.unlink()
(OUT / "status.json").write_text(json.dumps({"received": 0}))


def stored(event):
    dataset = event.dataset
    dataset.file_meta = event.file_meta
    target = OUT / f"{dataset.SOPInstanceUID}.dcm"
    dataset.save_as(target, enforce_file_format=True)
    files = list(OUT.glob("*.dcm"))
    (OUT / "status.json").write_text(json.dumps({"received": len(files)}))
    return 0x0000


def main():
    ae = AE(ae_title="E2E_RETURN")
    ae.add_supported_context(Verification)
    ae.add_supported_context(MRImageStorage)
    ae.start_server(("0.0.0.0", 11112), block=True, evt_handlers=[(evt.EVT_C_STORE, stored)])


if __name__ == "__main__":
    main()
