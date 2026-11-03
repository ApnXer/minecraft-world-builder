# Reference images and review renders

## Getting reference photos
- Wikimedia Commons serves any file directly: `https://commons.wikimedia.org/wiki/Special:FilePath/<File_name>.jpg?width=1280`
  (the media search page loads lazily — go straight to known file names or category pages).
- Open them in a browser/preview pane and screenshot. Write down: overall massing, number of storeys, roof forms and
  materials, tower caps, window rhythm, colour palette, what sits on what (e.g. castle on a crag, gatehouse in red brick).
- Photos are inspiration for proportions and materials; do not copy or redistribute them.

## Neuschwanstein notes (used in the example)
White limestone-like walls, grey-slate steep roofs, many slender round towers with cone spires, a stair tower, a
square tower with machicolated crown, red-brick gatehouse with stepped gable, rock crag with fir forest, curtain walls
and a steep approach. Build it as an irregular ensemble, not a symmetric block.

## Review renders (`scripts/render.py`)
- `Renderer(list(zip(mc.names, mc.props)))` then `ortho(arr, target=(x,y,z), yaw, pitch, width, height, ppb)` for
  aerials (ppb = pixels per block) and `persp(arr, eye, look, fov, width, height)` for player-eye views.
- Always include: overview, castle close-up, player-height street, market/plaza, courtyard, interior hall, forest,
  underside, plus a close-up of every new feature.
- Camera eye height for player views: ground + 2.6. Render with `aa=2`. Typical cost: 2–10 s per image.
- Simplifications: lanterns/pots are drawn as simple boxes; water/glass are blended; no in-game lighting.
