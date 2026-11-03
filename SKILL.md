---
name: minecraft-world-builder
description: Build large, high-quality Minecraft Java 1.21.11 structures and complete worlds from Python — castles, towns, forests, custom biomes — with coherent noise texturing, physical weathering, correct stair/wall/fence shapes, furnished interiors, a textured raytracer for visual review, and export to a ready-to-play world (Anvil regions + level.dat + biome datapack) or a Litematica schematic. Use when asked to design or generate big Minecraft builds, megabases, castles, villages, schematics (.litematic), or whole worlds, or to improve the visual quality of generated builds.
---

# Minecraft world builder

A numpy voxel toolkit plus the working method that produced a floating-island Neuschwanstein-style castle with a
town, market, 8 tree species and furnished interiors (see `examples/neuschwanstein/`, renders in `docs/renders/`).
The result must look **hand-built**, not generated. Most of the skill is the method in `references/quality-guide.md` —
read it before writing any build code.

## Workflow (do these in order)

1. **Pin the target version and read its jar.** Block ids, state names and textures come from the client jar
   (`scripts/mcpaths.py` finds it; `MC_JAR` overrides). Run `python scripts/extract_blocks.py` once. Never guess a
   block id or property value — `scripts/validate.py` checks the whole palette against the jar.
2. **Collect references.** Open real photos (Wikimedia Commons `Special:FilePath/<File>?width=1280` renders a file
   directly in a browser) and write down proportions, materials, roof forms, tower types before coding
   (`references/reference-images.md`).
3. **Lay out the site as 2D masks first** (terrain heights, plateau, streets, lots, water) — everything later is
   derived from them. Keep one coordinate convention: `arr[y, z, x]`, island centre `(C, C)`.
4. **Build in layers of detail:** terrain → structures (shell, relief, roofs, towers) → interiors → vegetation →
   ground cover → **post passes** (weathering, stair corner shapes, fence/wall/pane connections). The post passes
   run last and are what turns blocky geometry into believable architecture.
5. **Render and criticise, every iteration.** `scripts/render.py` is a textured raytracer using the jar's real
   textures/models with shadows. Render overview + a street-level view + a close-up of every new feature, find what
   looks wrong, fix, repeat. Do not stop after the first render that "works".
6. **Validate and export.** `scripts/validate.py` (block states vs jar) → `scripts/export_world.py` (playable world
   zip) or `scripts/npz_to_litematic.py` (schematic). Read the exported regions back and compare against the array.

## Toolkit map (all in `scripts/`)

| file | purpose |
|---|---|
| `mc.py` | world array `arr`, palette `B("block[state=..]")`, coherent noise (`noise3`, `fbm3`), `Mix` (noise-patch material mixes), `StairMix`, `fill/ball/disc/put`, 2D `fbm`, stair/slab/wall helpers |
| `arch.py` | architecture kit: `shell`, `facade` (plinth, pilasters, cornices, arched windows, sills, doors), height-field `roof`/`gable` (gable/hip/half-hip) + `dormer`, `round_tower` (corbels, crown, cone roof, finial), `spiral_stairs`, furniture (`table`, `bed`, `chandelier`, `floor`) |
| `houses.py` | randomised town-house kit (styles, timber patterns, jetties, wings, combined roofs, oriel, balcony, chimneys, interiors by trade) |
| `trees.py` | branching trees (oak, dark oak, willow, cherry, spruce, birch, pale oak), a giant mother tree, lianas, trunk vines, roots, fallen logs, shrubs |
| `post.py` | `weather()` (sun/shadow/damp/splash/streaks → clean/aged/dirty/mossy/cracked), `stair_shapes()` (inner/outer corners), `connections()` (fences, walls, panes) |
| `render.py` | numba raytracer with real textures; `Renderer(palette).ortho(...)/persp(...)` |
| `export_world.py` | build → Java world (regions, level.dat generated, biome datapack, block entities, zip) |
| `litematic_writer.py`, `npz_to_litematic.py` | build → `.litematic` |
| `nbt.py` | minimal typed NBT reader/writer |
| `validate.py`, `extract_blocks.py`, `mcpaths.py` | jar-based validation / block list / jar discovery |

Run the example: `cd examples/neuschwanstein && python build.py && python ../../scripts/export_world.py ...`
(exact commands in `README.md`). It needs numpy, scipy, numba, Pillow and a local Minecraft 1.21.11 install
(the jar is read, never copied or redistributed).

## Hard rules

- **No per-block random.** Use `Mix` / `noise3` / `fbm3` so materials form patches; use physical cues (shade, damp,
  height above ground, shelter, exposure) in `post.weather()` instead of dice rolls.
- **Roofs are height fields**, never stacked loops. Corner stair shapes and fence/wall connections are resolved by
  the post passes, so write straight stairs and let `post.stair_shapes()` fix the corners.
- **Vary everything**: no two houses alike; towers with different heights/caps; asymmetric layouts for "natural"
  scenes (symmetry only when the user wants it — then derive everything from `|x-C|`, `|z-C|`).
- **Everything walkable must be walkable**: doors, stairs that actually arrive at floors, no terrain blocking the
  route from gate to castle. Furnish interiors.
- **Never invent block ids/props.** Validate. Entity-rendered blocks (beds, chests) need explicit handling.
- **Honest limits:** schematics cannot carry biomes (use the datapack in `export_world.py`); the raytracer shows
  lanterns/pots as simplified boxes; in-game lighting and mob spawning were not tested.
