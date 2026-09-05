# From IMOD to mesh

Existing spines in this repo already have OBJ meshes, NFF/AZ files, and (for most) synpts. Use this page when **adding a new spine** from IMOD reconstructions.

You will need:

1. An IMOD model with **only** the toric-spine surface (`TS{id}.mod`)
2. Active zones, either as a separate IMOD model (`TS{id}_AZ.mod`) or as XYZ in the **same coordinate system as the mesh**
3. A neck location — prefer the automated mesh comparison below; a 3dmod estimate is a fallback

Then follow the [morphology pipeline](morphology.md) (skeletonize → fit SWC → sink → synpts).

## Surface mesh (`imod2obj`)

IMOD’s `imod2obj` converts a `.mod` file to a triangular `.obj` mesh. Run it from the command line (not the 3dmod GUI):

```bash
imod2obj.exe input.mod output.obj
```

Use the `.mod` that contains **only** the spine surface. If you convert the AZ model, IMOD tends to turn marker locations into spheres in the mesh.

Place the result at `data/mesh/TS{id}.obj`. Keep the full-cell mesh at `data/mesh/cell_wrapped_simplified.obj` if you will compute neckpoints.

## Active zones

From the AZ IMOD model in 3dmod: **File → Write as… NFF**. NFF is a dated but readable format; AZ coordinates appear as geometric primitives (often `s` lines). This repo reads them with `toric_spines_sim.utils.read_nff_s_points`.

```bash
uv run python scripts/active_zones_from_nff.py
```

That writes pixel AZ files under `data/pointsets/pixels/` and micron synapse sites `data/pointsets/microns/TS{id}_synpts.txt` (projected onto the matching pixel SWC, scaled by **5 nm/pixel**).

If you already have XYZ, put one coordinate per line in `data/pointsets/pixels/TS{id}_AZ.txt` (same frame as the mesh) and skip the NFF step. Do not mix pixel and micron frames.

## Neckpoint

The sink attaches at the neck: the center of the surface exposed when the spine was cut from the neuron. Exactness is less important than being on that cap.

**Supported path** in this repo: compare the isolated TS mesh to the full cell mesh.

```bash
uv run python scripts/compute_neckpoints.py TS1 --max-necks 1
```

That writes `data/pointsets/pixels/TS{id}_neckpoint.txt`. `append_sink.py` uses this file when present; otherwise it attaches at the SWC root.

You can still pick a point by eye in 3dmod if the automatic cap detection is wrong (multi-neck spines: TS3, TS21). Use `--dry-run` to inspect candidates without writing.
