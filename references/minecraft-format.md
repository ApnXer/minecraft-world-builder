# Minecraft 1.21.11 format notes (verified against the 1.21.11 client jar and a real save)

- **DataVersion 4671**, `level.dat` `version` field 19133, `Version.Name` "1.21.11", pack format `data_major 94`.
- **Block ids/states:** `assets/minecraft/blockstates/*.json` in the jar lists every block (1168). Property
  names/values come from `variants` keys and `multipart` `when` clauses. States absent there but valid on the block
  (waterlogged, distance, persistent, powered, snowy, age, level, …) must be allowed by hand (`validate.py`). Walls use
  `none/low/tall` for sides plus `up`; fences/panes/bars use booleans. Beds, chests, decorated pots are entity-rendered
  (no variants) but still have `facing`/`part`.
- **New in this version range worth using:** resin bricks (+stairs/slabs/walls), tuff bricks/polished tuff,
  pale oak, pale moss + hanging moss, leaf litter, wildflowers, bush, firefly bush, eyeblossoms, copper lanterns,
  copper chains (`copper_chain`; `iron_chain` is the iron one), mud bricks, cherry wood. `chain` was renamed.
- **Chunk (Anvil, zlib/compression type 2):** root has `DataVersion`, `xPos`, `zPos`, `yPos` (-4), `Status`
  `minecraft:full`, `sections` (24, Y −4…19), `block_entities`, `Heightmaps` optional, `isLightOn` 0 (the game relights).
  Each section: `Y` (byte), `block_states{palette, data?}`, `biomes{palette}`. Data entries are packed **without
  spanning longs** (`per = 64 // bits`, `bits = max(4, ceil(log2(n)))`); a single-entry palette has no `data`.
  Order inside a section: index = (y·16 + z)·16 + x. Region header: 1024 offsets (sector<<8 | count) + 1024 timestamps.
- **Block entities** must exist for chests, barrels, beds, furnaces, campfires, lecterns, bells, … or the blocks are
  broken in game. Minimal tag: `id`, `x`, `y`, `z`, `keepPacked`.
- **level.dat:** generated in `export_world.py`. Flat overworld generator with `settings.biome` = custom biome and
  one air layer gives a void world that uses the datapack biome everywhere. `generate_features: 0`.
- **Custom biome datapack (1.21.11 format):** `datapacks/<id>/pack.mcmeta` with `min_format/max_format` `[94, 0]`,
  biome at `data/<ns>/worldgen/biome/<name>.json`. **Visuals moved into `attributes`** (e.g.
  `minecraft:visual/sky_color`, `fog_color`, `water_fog_color`, `ambient_particles`, `minecraft:audio/background_music`);
  `effects` keeps `grass_color`, `foliage_color`, `dry_foliage_color`, `water_color`. Enable in level.dat
  `DataPacks.Enabled` as `file/<id>`.
- **Litematica** (`.litematic`): gzip NBT, `Version` 6, `MinecraftDataVersion`, region with `BlockStatePalette`
  (index 0 = air) and tightly packed `BlockStates` long array (entries may span longs, `bits = max(2, ceil(log2(n)))`),
  order (y·sizeZ + z)·sizeX + x. Litematica cannot store biomes. `scripts/litematic_writer.py` implements it.
- **Vanilla structure `.nbt`** has a 48³ limit only for structure blocks; Litematica loads larger files.
