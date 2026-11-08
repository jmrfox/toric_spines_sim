"""Inject ring_data.json into template.html -> loop_closure.html."""
from pathlib import Path

here = Path(__file__).parent
data = (here / "ring_data.json").read_text()
assert "</script" not in data
page = (here / "template.html").read_text().replace("__RING_DATA__", data)
(here / "loop_closure.html").write_text(page)
print("wrote", here / "loop_closure.html", f"{len(page) / 1e6:.2f} MB")
