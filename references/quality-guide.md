# Quality guide — what made the builds look right (and what looked wrong)

Lessons from iterating with a user who kept saying "random noise", "buggy roof edges", "needs more depth/detail",
"too symmetric", "all houses look alike". Each section: the rule, the technique, the failure it prevents.

## 1. Texture: coherent noise + physics, not dice
- **Failure:** per-block hash picking from a block list ("confetti"). Looks like a bad texture pack.
- **Material mixes** (`mc.Mix`): 3D value-noise `fbm3` mapped through an error function to a uniform value, then
  thresholded by cumulative weights. `scale` = patch size in blocks. Add a faint 1.6-block noise term so patch
  borders are ragged. Stretch one axis (`stretch=(1,5,1)`) for vertical streaks.
- **Weathering** (`post.weather`) decides per block from measurable features:
  - sun/shadow: ray towards the sun through solid blocks (leaves give partial shade; pure north faces get none)
  - damp: north-facing, shaded, low (splash zone ≤5 blocks above ground) → mossy/deep/cobbled variants
  - shelter under overhangs → dirty/sooty (darker, no moss); rain streaks: an opening above in the same column
  - exposed edges/corners + noise → cracked; batches/patina from a low-frequency `fbm3`
  - each material family has clean / aged / dirty / mossy / deep / cracked variants and keeps stair/slab/wall kinds
- Timber: sun-bleached (stripped) where sunlit, dark where shaded.
- **Ground:** forest floor follows a canopy-shade map (podzol/moss/coarse dirt under trees, grass in sun); plants in
  clumps from noise, not scattered singles. Rock uses sedimentary **strata** (warped bands, calcite seams, darker with
  depth), not random stone types.

## 2. Roofs: height fields
- **Failure:** roofs built as loops of stairs → jagged eaves, gaps at gable ends, wrong hips/valleys.
- Define `H(x, z)` per roof part (gable: distance to ridge; hip: min of both distances; half-hip: min(gable,
  clipped hip)). Combine parts with `max` so L-shaped houses get real valleys. For each cell place a stair facing the
  neighbour with the larger height (up-slope); ridge cells get a slab cap; a solid body fills underneath; gable end
  walls are filled to the roof surface; verge gets upside-down stairs under the overhang.
- Resolve stair corners afterwards with `post.stair_shapes()` (Minecraft's `StairBlock` rules). A dormer is a tiny
  height-field gable whose ridge is perpendicular to the main ridge.

## 3. Depth on walls
Flat walls look generated. Per facade: stone plinth proud by 1 with stair chamfer; pilasters at bay edges; cornice
courses at every storey from upside-down stairs; windows recessed one block (glass pane behind the wall line) with
sill slab, arched head (stairs facing inward) and hood moulding; quoins alternating long/short stones at corners;
jetty (upper storey overhangs on upside-down stairs); shutters as open trapdoors; flower boxes (trapdoor + pot);
door canopies; downspouts; drip-edge gravel strip; firewood stacks, barrels, composters.

## 4. Towers
Round towers: corbelled base, string courses, window columns at angles (`cos(n·angle) > 0.82`), projecting
machicolation ring + crenels with slab caps, cone roof from stairs with flared eave, gold finial + lightning rod,
floors + a spiral staircase whose steps land at the floor heights (offset angle so the step at floor level faces the
door). Vary height, radius, cone height and cap (cone / hip / crown) — never place the same tower twice.

## 5. Variety rules for houses (`houses.House`)
Style (timber/stone/plaster/brick), 1–4 storeys, 4–5 block storey height, timber frame patterns (plain/diag/X/K)
with panel-based windows (windows sit between posts, never on them), jetty yes/no, L-wing yes/no, roof kind
(gable/half-hip/hip), rise 1 or 2, ridge parallel or gable to the street, 7 roof materials, 6 plaster tints,
8 shutter colours, 8 door woods, door canopy or lantern only, optional oriel/balcony, 1–2 chimneys at the real roof
height (read from the roof height map), interior by trade (baker, smith, weaver, scholar, brewer, cartographer,
butcher, tailor, home). Place lots along streets with a free-footprint test; draw a worn footpath door → street.

## 6. Streets
Derive from a distance transform of the street mask: gutter (damp mossy cobble, gravel, iron-trapdoor drains, weeds)
→ faint wheel ruts on wide streets → worn crown (polished andesite/stone) broken by wear noise → repair patches and
mud/water puddles from low-frequency noise → dressed-stone curb where street meets grass. Market squares: concentric
courses around the fountain with chiseled/polished accents and wear lines.

## 7. Trees (`trees.py`)
Not blobs on sticks: tapered oriented-log trunks (axis follows branch direction), flared roots, 3–5 primary limbs +
secondary twigs, lumpy leaf clusters with holes (ellipsoids with noisy radius, 6 % hollow), hanging lianas
(`cave_vines` with berries, `pale_hanging_moss`), vines/glow lichen on bark, leaf litter, fallen logs, shrubs.
Species by zone noise: willow near water, cherry/pale/spruce/birch/oak groves elsewhere; one colossal mother tree.
Leaves must be `persistent=true`.

## 8. Layout & walkability
Natural scenes are asymmetric; real castles (Neuschwanstein) are an irregular ensemble on a crag: palas, stair tower,
square tower, wings, curtain walls, gatehouse, forecourt, ramp from the town. Check the route gate → town → ramp →
gatehouse → courtyard → doors → stairs with a player-height camera render. Keep a clear road: no terrain/trees in it.

## 9. Interiors
Floors with beam undersides, partitions with doors, chandeliers (chain + lantern), beds with matching carpets,
tables with stair chairs, bookshelves, kitchen with furnaces/smokers/cauldrons, throne hall with columns, gallery,
dais, glazed-terracotta carpet border; guard rooms, smithies, library with lecterns/enchanting table.

## 10. Terrain
Large-scale `fbm` heights, lakes where height < water level, a stream carved along a meandering polyline that ends
as a waterfall, rounded-rectangle crag platform with noisy steep flanks, floating island underside with strata,
dripstone, hanging roots and glow vines; `slope` map drives rock vs. grass and cliff vines.

## Common bugs hit (fix once, remember)
- numpy: `np.where(mask_array[...] < x, ...)` on already-masked arrays → index the full array, then mask the result.
- Leaves and other blocks that are not solid for shadow casting must be handled separately (`solid_table`).
- Palette growth: `Mix`/`B()` register blocks lazily — validate the final palette, not the code.
- Always re-render after a "small" change; roof/stair mistakes only show in renders.
