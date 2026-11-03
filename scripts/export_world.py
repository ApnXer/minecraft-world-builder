"""Write a build (npz of palette ids + palette json) as a complete Minecraft 1.21.11 Java world.

Output: Anvil region files, level.dat (generated here, no template needed), a flat void world that uses a
custom-biome datapack, block entities for chests/barrels/beds/..., icon.png, and a ready-to-unzip .zip.

    python export_world.py --npz build.npz --palette build_palette.json --name "My World" --spawn 0,81,60

Array layout: arr[y, z, x]; the array centre (S // 2) maps to world x = z = 0; build y = 0 maps to world y = --yoff.
"""
import argparse
import json
import os
import shutil
import struct
import time
import zipfile
import zlib

import numpy as np

import nbt

DATA_VERSION = 4671          # Minecraft 1.21.11
SERIES_ID = 19133            # level.dat "version" field of 1.21.11

BE_IDS = {"chest": "chest", "trapped_chest": "trapped_chest", "barrel": "barrel", "campfire": "campfire",
          "soul_campfire": "campfire", "lectern": "lectern", "bell": "bell", "beehive": "beehive", "bee_nest": "beehive",
          "decorated_pot": "decorated_pot", "chiseled_bookshelf": "chiseled_bookshelf", "enchanting_table": "enchanting_table",
          "furnace": "furnace", "smoker": "smoker", "blast_furnace": "blast_furnace", "brewing_stand": "brewing_stand",
          "jukebox": "jukebox", "hopper": "hopper", "dispenser": "dispenser", "dropper": "dropper"}

# vanilla 1.21.11 game rule defaults (int rules are Int tags, boolean rules are Byte tags)
GAME_RULES = {
    "spawn_wandering_traders": 1, "block_drops": 1, "reduced_debug_info": 0, "show_death_messages": 1, "spawn_monsters": 1,
    "spawner_blocks_work": 1, "tnt_explodes": 1, "immediate_respawn": 0, "player_movement_check": 1, "spread_vines": 1,
    "block_explosion_drop_decay": 1, "max_entity_cramming": 24, "forgive_dead_players": 1, "fall_damage": 1,
    "send_command_feedback": 1, "global_sound_events": 1, "elytra_movement_check": 1, "fire_spread_radius_around_player": 128,
    "freeze_damage": 1, "mob_explosion_drop_decay": 1, "natural_health_regeneration": 1,
    "players_nether_portal_default_delay": 80, "mob_drops": 1, "log_admin_commands": 1, "mob_griefing": 1, "pvp": 1,
    "spawn_mobs": 1, "spectators_generate_chunks": 1, "players_nether_portal_creative_delay": 0,
    "max_command_sequence_length": 65536, "players_sleeping_percentage": 100, "advance_weather": 1,
    "max_block_modifications": 32768, "max_command_forks": 65536, "drowning_damage": 1, "show_advancement_messages": 1,
    "command_block_output": 1, "locator_bar": 1, "respawn_radius": 10, "raids": 1, "spawn_phantoms": 1,
    "max_snow_accumulation_height": 1, "limited_crafting": 0, "allow_entering_nether_using_portals": 1,
    "lava_source_conversion": 0, "tnt_explosion_drop_decay": 0, "keep_inventory": 0, "universal_anger": 0,
    "spawn_patrols": 1, "random_tick_speed": 3, "fire_damage": 1, "entity_drops": 1, "advance_time": 1,
    "command_blocks_work": 1, "spawn_wardens": 1, "projectiles_can_break_blocks": 1, "water_source_conversion": 1,
    "ender_pearls_vanish_on_death": 1,
}


class Exporter:
    def __init__(self, npz, palette, name, biome, yoff, pack_id="build_biome"):
        self.arr = np.load(npz)["arr"]
        self.pal = json.load(open(palette))
        self.name, self.biome, self.yoff, self.pack_id = name, biome, yoff, pack_id
        self.Yb, self.S, _ = self.arr.shape
        self.C = self.S // 2
        self.pal_tags = []
        for nm, pr in self.pal:
            body = {"Name": nbt.st(nm)}
            if pr:
                body["Properties"] = nbt.cd({k: nbt.st(v) for k, v in pr.items()})
            self.pal_tags.append(nbt.cd(body))
        self.be_of = np.full(len(self.pal), -1, np.int32)
        self.be_names = []
        for i, (nm, _) in enumerate(self.pal):
            short = nm.replace("minecraft:", "")
            be = "bed" if short.endswith("_bed") else BE_IDS.get(short)
            if be:
                self.be_of[i] = len(self.be_names)
                self.be_names.append("minecraft:" + be)

    @staticmethod
    def pack(vals, bits):
        per = 64 // bits                                     # since 1.16 entries never span two longs
        v = vals.astype(np.uint64)
        pad = (-len(v)) % per
        if pad:
            v = np.concatenate([v, np.zeros(pad, np.uint64)])
        v = v.reshape(-1, per) << (np.arange(per, dtype=np.uint64) * np.uint64(bits))
        return np.bitwise_or.reduce(v, axis=1).view(np.int64)

    def section(self, blocks, sy):
        uniq, inv = np.unique(blocks.reshape(-1), return_inverse=True)
        bs = {"palette": nbt.lst(nbt.COMP, [self.pal_tags[u] for u in uniq])}
        if len(uniq) > 1:
            bs["data"] = nbt.la(self.pack(inv, max(4, int(len(uniq) - 1).bit_length())))
        return nbt.cd({"Y": nbt.b(sy), "block_states": nbt.cd(bs),
                       "biomes": nbt.c(palette=nbt.lst(nbt.STR, [nbt.st(self.biome)]))})

    def chunk(self, cx, cz):
        C, S, Yb = self.C, self.S, self.Yb
        x0, z0 = cx * 16 + C, cz * 16 + C
        secs, bents, nonempty = [], [], False
        for sy in range(-4, 20):
            wy0 = sy * 16
            blk = np.zeros((16, 16, 16), np.uint16)
            by0 = wy0 - self.yoff
            ya, yb = max(by0, 0), min(by0 + 16, Yb)
            xa, xb, za, zb = max(x0, 0), min(x0 + 16, S), max(z0, 0), min(z0 + 16, S)
            if ya < yb and xa < xb and za < zb:
                blk[ya - by0:yb - by0, za - z0:zb - z0, xa - x0:xb - x0] = self.arr[ya:yb, za:zb, xa:xb]
                nonempty |= bool(blk.any())
                for yy, zz, xx in zip(*np.nonzero(self.be_of[blk] >= 0)):
                    bents.append(nbt.c(id=nbt.st(self.be_names[self.be_of[blk[yy, zz, xx]]]), x=nbt.i(cx * 16 + xx),
                                       y=nbt.i(wy0 + yy), z=nbt.i(cz * 16 + zz), keepPacked=nbt.b(0)))
            secs.append(self.section(blk, sy))
        tag = nbt.cd({
            "DataVersion": nbt.i(DATA_VERSION), "xPos": nbt.i(cx), "zPos": nbt.i(cz), "yPos": nbt.i(-4),
            "Status": nbt.st("minecraft:full"), "LastUpdate": nbt.l(0), "InhabitedTime": nbt.l(0), "isLightOn": nbt.b(0),
            "sections": nbt.lst(nbt.COMP, secs), "block_entities": nbt.lst(nbt.COMP, bents),
            "block_ticks": nbt.lst(nbt.COMP, []), "fluid_ticks": nbt.lst(nbt.COMP, []),
            "PostProcessing": nbt.lst(nbt.LIST, [nbt.lst(nbt.SHORT, []) for _ in range(24)]),
            "structures": nbt.c(References=nbt.c(), starts=nbt.c()),
        })
        return tag, nonempty

    def write_regions(self, world):
        os.makedirs(os.path.join(world, "region"), exist_ok=True)
        lo, hi = (0 - self.C) // 16, (self.S - 1 - self.C) // 16
        regions, count = {}, 0
        for cx in range(lo, hi + 1):
            for cz in range(lo, hi + 1):
                tag, nonempty = self.chunk(cx, cz)
                regions.setdefault((cx >> 5, cz >> 5), []).append((cx, cz, zlib.compress(nbt.dumps(tag), 6)))
                count += nonempty
        now = int(time.time())
        for (rx, rz), chunks in regions.items():
            header, body, sector = bytearray(8192), bytearray(), 2
            for cx, cz, data in chunks:
                payload = struct.pack(">IB", len(data) + 1, 2) + data
                payload += b"\x00" * ((-len(payload)) % 4096)
                nsec = len(payload) // 4096
                k = (cx & 31) + (cz & 31) * 32
                header[4 * k:4 * k + 4] = struct.pack(">I", (sector << 8) | nsec)
                header[4096 + 4 * k:4096 + 4 * k + 4] = struct.pack(">I", now)
                body += payload
                sector += nsec
            with open(os.path.join(world, "region", f"r.{rx}.{rz}.mca"), "wb") as fh:
                fh.write(header + body)
        return count, len(regions)

    def level_dat(self, world, spawn, seed=0):
        int_rules = {k for k, v in GAME_RULES.items() if v not in (0, 1)} | {"players_nether_portal_creative_delay",
                                                                              "max_snow_accumulation_height"}
        rules = {f"minecraft:{k}": (nbt.i(v) if k in int_rules else nbt.b(v)) for k, v in GAME_RULES.items()}
        flat = nbt.c(type=nbt.st("minecraft:flat"), settings=nbt.c(
            biome=nbt.st(self.biome), features=nbt.b(0), lakes=nbt.b(0),
            layers=nbt.lst(nbt.COMP, [nbt.c(block=nbt.st("minecraft:air"), height=nbt.i(1))]),
            structure_overrides=nbt.lst(nbt.STR, [])))
        dims = nbt.c(**{
            "minecraft:overworld": nbt.c(type=nbt.st("minecraft:overworld"), generator=flat),
            "minecraft:the_nether": nbt.c(type=nbt.st("minecraft:the_nether"), generator=nbt.c(
                type=nbt.st("minecraft:noise"), settings=nbt.st("minecraft:nether"),
                biome_source=nbt.c(type=nbt.st("minecraft:multi_noise"), preset=nbt.st("minecraft:nether")))),
            "minecraft:the_end": nbt.c(type=nbt.st("minecraft:the_end"), generator=nbt.c(
                type=nbt.st("minecraft:noise"), settings=nbt.st("minecraft:end"),
                biome_source=nbt.c(type=nbt.st("minecraft:the_end")))),
        })
        data = nbt.cd({
            "DataVersion": nbt.i(DATA_VERSION), "version": nbt.i(SERIES_ID),
            "Version": nbt.c(Id=nbt.i(DATA_VERSION), Name=nbt.st("1.21.11"), Series=nbt.st("main"), Snapshot=nbt.b(0)),
            "LevelName": nbt.st(self.name), "GameType": nbt.i(1), "Difficulty": nbt.b(1), "DifficultyLocked": nbt.b(0),
            "hardcore": nbt.b(0), "allowCommands": nbt.b(1), "initialized": nbt.b(1), "WasModded": nbt.b(0),
            "ServerBrands": nbt.lst(nbt.STR, [nbt.st("vanilla")]),
            "Time": nbt.l(0), "DayTime": nbt.l(6000), "LastPlayed": nbt.l(int(time.time() * 1000)),
            "raining": nbt.b(0), "rainTime": nbt.i(100000), "thundering": nbt.b(0), "thunderTime": nbt.i(100000),
            "clearWeatherTime": nbt.i(0),
            "WanderingTraderSpawnChance": nbt.i(25), "WanderingTraderSpawnDelay": nbt.i(22800),
            "CustomBossEvents": nbt.c(), "ScheduledEvents": nbt.lst(nbt.COMP, []),
            "DragonFight": nbt.c(NeedsStateScanning=nbt.b(1), DragonKilled=nbt.b(0), PreviouslyKilled=nbt.b(0),
                                 Gateways=nbt.lst(nbt.INT, [nbt.i(g) for g in
                                                            np.random.default_rng(seed).permutation(np.arange(1, 21))])),
            "game_rules": nbt.cd(rules),
            "spawn": nbt.c(pos=nbt.ia(spawn), pitch=nbt.f(0.0), yaw=nbt.f(180.0), dimension=nbt.st("minecraft:overworld")),
            "DataPacks": nbt.c(Enabled=nbt.lst(nbt.STR, [nbt.st("vanilla"), nbt.st(f"file/{self.pack_id}")]),
                               Disabled=nbt.lst(nbt.STR, [nbt.st(s) for s in ("minecart_improvements", "redstone_experiments",
                                                                              "trade_rebalance")])),
            "WorldGenSettings": nbt.c(seed=nbt.l(seed), generate_features=nbt.b(0), bonus_chest=nbt.b(0), dimensions=dims),
        })
        nbt.save(os.path.join(world, "level.dat"), nbt.c(Data=data))

    def datapack(self, world):
        base = os.path.join(world, "datapacks", self.pack_id)
        ns, bname = self.biome.split(":")
        bdir = os.path.join(base, "data", ns, "worldgen", "biome")
        os.makedirs(bdir, exist_ok=True)
        json.dump({"pack": {"description": f"{self.name}: custom biome", "min_format": [94, 0], "max_format": [94, 1]}},
                  open(os.path.join(base, "pack.mcmeta"), "w"), indent=2)
        biome = {
            "attributes": {
                "minecraft:visual/sky_color": "#7fc4ff", "minecraft:visual/fog_color": "#c8ecdd",
                "minecraft:visual/water_fog_color": "#1d6f6a",
                "minecraft:visual/ambient_particles": [{"particle": {"type": "minecraft:spore_blossom_air"}, "probability": 0.004}],
                "minecraft:audio/background_music": {"default": {"max_delay": 24000, "min_delay": 12000,
                                                                 "sound": "minecraft:music.overworld.cherry_grove"}},
            },
            "carvers": [], "downfall": 0.8,
            "effects": {"grass_color": "#5fd18a", "foliage_color": "#4cc27a", "dry_foliage_color": "#9cbf7a",
                        "water_color": "#3fd0c9"},
            "features": [], "has_precipitation": True, "spawn_costs": {},
            "spawners": {"creature": [{"type": "minecraft:sheep", "weight": 8, "minCount": 2, "maxCount": 4},
                                      {"type": "minecraft:rabbit", "weight": 4, "minCount": 2, "maxCount": 3},
                                      {"type": "minecraft:fox", "weight": 2, "minCount": 1, "maxCount": 2}],
                         "ambient": [], "monster": [], "axolotls": [], "misc": [], "underground_water_creature": [],
                         "water_creature": [], "water_ambient": []},
            "temperature": 0.7,
        }
        json.dump(biome, open(os.path.join(bdir, f"{bname}.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="build.npz")
    ap.add_argument("--palette", default="build_palette.json")
    ap.add_argument("--name", default="Generated World")
    ap.add_argument("--biome", default="build:custom_biome", help="namespace:name of the datapack biome")
    ap.add_argument("--yoff", type=int, default=40, help="world y of build y=0")
    ap.add_argument("--spawn", default="0,81,0", help="world x,y,z of the spawn point")
    ap.add_argument("--icon", default="icon.png")
    ap.add_argument("--out", default="world_out")
    a = ap.parse_args()
    t0 = time.time()
    ex = Exporter(a.npz, a.palette, a.name, a.biome, a.yoff)
    shutil.rmtree(a.out, ignore_errors=True)
    world = os.path.join(a.out, a.name)
    os.makedirs(world)
    n_chunks, n_regions = ex.write_regions(world)
    ex.level_dat(world, [int(v) for v in a.spawn.split(",")])
    ex.datapack(world)
    if os.path.exists(a.icon):
        shutil.copyfile(a.icon, os.path.join(world, "icon.png"))
    zpath = os.path.join(a.out, a.name.replace(" ", "_") + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(world):
            for fn in fs:
                full = os.path.join(dp, fn)
                zf.write(full, os.path.relpath(full, a.out))
    print(f"chunks with blocks: {n_chunks}, regions: {n_regions}, zip {os.path.getsize(zpath) / 1e6:.1f} MB, "
          f"{time.time() - t0:.1f}s -> {zpath}")


if __name__ == "__main__":
    main()
