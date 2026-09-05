# Glossary

**SWC.** File format for cable morphologies ([spec](https://swc-specification.readthedocs.io/en/latest/index.html)). Each row is `index tag x y z r parent`. A strict SWC describes an [arborescence](https://en.wikipedia.org/wiki/Arborescence_%28graph_theory%29) (acyclic rooted directed tree).

**Cable model.** A 3D graph of neuronal (or subneuronal) morphology: vertices have XYZ and radius; edges are [frusta](https://en.wikipedia.org/wiki/Frustum). Not synonymous with SWC — a cable model need not be rooted, directed, or acyclic. This project writes cables as SWC **plus header directives** (`# CYCLE_BREAK`, `# MULTI_NECK`) so loops can be restored in Arbor as gap junctions.

**Skeleton.** A 1D curve network in 3D, stored as **polylines** (`data/skeletons/TS{id}.polylines.txt`). Convertible to an undirected graph. Input to mascaf.

**Triangle mesh.** Here, a [triangle mesh](https://en.wikipedia.org/wiki/Triangle_mesh), usually Wavefront **OBJ**. For MCFS it should be a single component and [watertight](https://davidstutz.de/a-formal-definition-of-watertight-meshes/) (no holes in the surface, no duplicate vertices/edges/faces, no self-intersections). It **may** contain [topological holes](https://en.wikipedia.org/wiki/Genus_(mathematics)) (genus > 0) — that is the point of toric spines.

**MorphologyGraph / basis.** mascaf terms for the central graph of a cable **without radii**: a downsampled skeleton that will become SWC nodes and edges. **Basis optimization** refines that graph before radius fitting.

**Morphology vs geometry vs topology** (as used here):

- *Morphology* — the biologically relevant cable in the simulation
- *Geometry* — meshes, skeletons, 3D graphs (MCFS is a geometric algorithm)
- *Topology* — closed surfaces and their holes → **genus**

**Sink.** A cylindrical cable appended at the spine neck (SWC tags 5 / 6) so the isolated spine can leak into a stand-in for the rest of the cell.

**Neckpoint.** A point near the center of the cut surface where the spine was separated from the dendrite. Used as the sink attachment. Computed by comparing the isolated mesh to `cell_wrapped_simplified.obj` (`scripts/compute_neckpoints.py`).

**Active zone (AZ) / synpts.** Synapse coordinates. NFF exports from IMOD become `TS{id}_AZ.txt`; projected, micron-scaled sites used in simulation are `TS{id}_synpts.txt`.

**SSN.** Space-specific neuron in barn-owl ICx (external nucleus of the inferior colliculus) — the cell class that bears toric spines.

**Neurosignature.** Framework that embeds a multi-channel event-in / multi-channel signal-out operator \(F\) in a descriptor space \(\mathcal{Z}\) under an input ensemble \(\theta\). The current \(W\) compares internal activity to a downstream output (residuals, residual energy, transfer efficiency); users can add descriptors. See [Neurosignature](neurosignature.md).
