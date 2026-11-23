# Third-party assets

anybot's code is MIT licensed. It ships **no** third-party models, scans or weights: everything below is
downloaded on first use to `~/.anybot` (or by `robot_descriptions` to its own cache) and keeps its own licence.
Robot and company names are used only to say which model is loaded; no endorsement is implied.

| What | Source | Licence |
|---|---|---|
| Robot models (Panda, UR5e, xArm7, SO-100, Spot, Go2, ANYmal C, G1, Tidybot, ...) | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) via [robot_descriptions](https://github.com/robot-descriptions/robot_descriptions.py) | per model (mostly Apache-2.0 / BSD-3 / MIT), see each model folder |
| Scanned household objects in the `lab` world | [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) via [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) | CC BY 4.0 |
| G1 walking policy | [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | BSD-3-Clause |
| `apartment` world scan (`apartment_1`) | [Habitat test scenes](https://huggingface.co/datasets/ai-habitat/habitat_test_scenes), from the [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) (Meta) | **CC BY-NC 4.0: non-commercial use only** |
| Physics | [MuJoCo](https://github.com/google-deepmind/mujoco) | Apache-2.0 |
| UI font | [Geist](https://vercel.com/font) (loaded from a CDN) | SIL OFL 1.1 |

If you use anybot commercially, replace the `apartment` scan with your own (see `anybot import-scene`).
