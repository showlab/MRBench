# Credits

No asset is distributed with MRBench. The following are fetched or loaded on your machine at run time and used
under their own licences; pictures and videos rendered with them must carry this credit.

| what | source | licence | how it gets here |
|---|---|---|---|
| Franka Emika Panda USD | NVIDIA Isaac Sim assets (`Robots/Franka/franka.usd`) with Isaac Lab `FRANKA_PANDA_HIGH_PD_CFG`'s drives | NVIDIA Isaac Sim asset terms | loaded from NVIDIA's asset server |
| AgileX PiPER URDF + meshes | [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) `src/piper_description` @ `ac41fcbcdda598f01b51cf6175ed9a24d0dacadc` | MIT | `mrbench assets fetch piper` |
| SO-101 URDF + meshes | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101` @ `a758567c3978dfeefe282ede0500085a48fe8f78` | Apache-2.0 | `mrbench assets fetch so101` |
| The room, the table legs' material | NVIDIA Isaac Sim assets (`Environments/Simple_Room/simple_room.usd`, `NVIDIA/Materials/Base/Metals/Aluminum_Cast.mdl`) | NVIDIA Isaac Sim asset terms | loaded from NVIDIA's asset server |
| YCB props (21 scans) | the YCB object and model set (Calli et al.), as NVIDIA ships it (`Props/YCB/Axis_Aligned/*.usd`) | YCB: CC BY 4.0; NVIDIA's files: Isaac Sim asset terms | loaded from NVIDIA's asset server |
| Table and floor textures | [Poly Haven](https://polyhaven.com): `oak_veneer_01`, `ash_veneer`, `plywood`, `marble_01`, `rough_linen`, `concrete_floor_01`, `laminate_floor_02`, `herringbone_parquet`, `dirty_carpet`, `large_grey_tiles`, `terrazzo_tiles` (2k, pinned by SHA-256) | CC0 | `mrbench assets looks` |
| Studio light | Poly Haven HDRI `brown_photostudio_02` (4k, pinned by SHA-256) | CC0 | `mrbench assets looks` |

Built on [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) and
[Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause).
