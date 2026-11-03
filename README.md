# minecraft-world-builder

An AI skill (`SKILL.md`) plus Python tools for generating large, high-quality Minecraft Java **1.21.11** builds:
castles, towns, forests and custom biomes, exported as a playable world or a Litematica schematic.

![overview](docs/renders/overview.png)

## What is in here
- `SKILL.md` — the skill: workflow, toolkit map, hard rules. Start here (also for AI agents).
- `references/` — quality guide (what makes builds look hand-made), 1.21.11 format notes, reference-image workflow.
- `scripts/` — the toolkit (voxel array + coherent noise, architecture kit, houses, trees, weathering post-pass,
  textured raytracer, world/schematic exporters, validator).
- `examples/neuschwanstein/` — a complete build (floating island, castle on a crag, town with market, forests).
- `docs/renders/` — renders produced by the raytracer from the example.

## Requirements
Python 3.10+, `pip install -r requirements.txt`, and a local Minecraft **1.21.11** install (start that version once).
The tools read block ids, models and textures from the client jar; it is never copied into this repo.
Set `MC_JAR=/path/to/1.21.11.jar` if it is not in the default `.minecraft` location.

## Quick start
```bash
python scripts/extract_blocks.py                      # list of all block ids of the jar
cd examples/neuschwanstein
python build.py                                       # ~1 min -> build.npz + build_palette.json
python views.py overview castle north_street          # raytraced review images -> renders/
python ../../scripts/validate.py                      # block states vs the jar
python ../../scripts/export_world.py --name "Neuschwanstein" --biome build:custom_biome \
       --yoff 40 --spawn 56,81,86 --out world_out     # playable world zip
python ../../scripts/npz_to_litematic.py --out build.litematic   # optional schematic (no biome)
```
Unzip the world into `.minecraft/saves`. Not tested in game by the author; renders approximate lighting.

## License
MIT
