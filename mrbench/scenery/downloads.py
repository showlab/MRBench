"""Photographs the looks are made of, fetched once into ``third_party/looks/`` and checked by SHA-256.

Every file is CC0, from Poly Haven: 2k texture sets (a diffuse, a normal and a roughness map each) for the
table tops and the floors, and a photo studio's HDRI that lights the scene.  Nothing here is committed (see
NOTICE); ``mrbench assets looks`` fetches it all ahead of time, and an Isaac run fetches what it uses.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

from ..core.errors import ConfigError
from ..core.paths import third_party_dir

POLY = "https://dl.polyhaven.org/file/ph-assets"
#: Poly Haven refuses Python's default user agent
USER_AGENT = "MultiRobotBench-asset-prep/1.0 (research benchmark)"
#: the maps of a texture set, in this order
MAPS = ("diff", "nor_gl", "rough")
#: texture set -> the SHA-256 of its diffuse, normal and roughness maps (2k JPEG)
TEXTURES: dict[str, tuple[str, str, str]] = {
    "oak_veneer_01": ("0718c2e1a209afe6138ccb41d70e1a18cd825d1e73db33ad7cf80806fa9ce5a4",
                      "e74a9d3ddcd86473a21815a28464f4dac4ceb8f86e7f170226dede8ed70d8eb9",
                      "5c81718ee172299c44e0754fadbe652758b6feed05a8fd88be6efff5aa70ca1e"),
    "ash_veneer": ("00abb70f86bdfe49aa6e09f350af9e8e1605ccf13171f9be46ceb6018846c06b",
                   "64bbd775ba2978aa12108ecd58419c333eb0c194b52cbc8e1d2bea02252020af",
                   "3e6b077f6af6ba7725668b438bfc523f1342d0fb4bd497f1c9668663ad189ef0"),
    "plywood": ("050a15a796413b91c9cd5ca026dd3c4df83746bb1103ae14ae91503577dead8c",
                "2391dbe49aaecd6584d12731d4470bf91ba33b5c7f05c78ec5900fcae28f0310",
                "bfd6947bca45a306f8c29add227c5eb9c7903d0cf1e72ce114483eb75305aa74"),
    "marble_01": ("d403786171716f86718bdd67eba923d4fb6125c0636bacef0e6a21dd5d623a48",
                  "d5e17ccb2913adbf28fcb781fddf0aa711259ddea6c1c918442f9a3589aa4660",
                  "1b970e033856c93ee7390d947da5340e101b489fc8c1463354ea9e6655ce039a"),
    "rough_linen": ("e71016e08dfebea1dd4e8a2a675e3b617a889a782af782f1abb795906f4db1b4",
                    "c5733ad50cfe0f90c546f3d4076590c1c78a4fd1a83c967cc9c92dc8175b99e6",
                    "48400d48fa203882bf5d984d5beb2f95edb1550309924601c018f2c4f3f80731"),
    "concrete_floor_01": ("ca24c52dabae67ef49093f91d8f9b9e5328a3b9fcb1327742ff6181ee542e017",
                          "796b010208e346bbeec4c468f286c3892852a454f77cb516d6f4a6e7c240de01",
                          "e66fa941d6daa33ba334c853bf126c85f1d77e7b72778073237e4ef8c7c97774"),
    "laminate_floor_02": ("2a1e6519309b369f749f06aaf241ac67ea762c686b6a848c1a79331b18a30521",
                          "49aa58c34c6770d209e63f9bdb4d16326f5dab24474cfeb18433e63fc54456de",
                          "f547988692d1b53274b3023e9ed89d8280087a5ff1018cd64f83e77440ab54c4"),
    "herringbone_parquet": ("8305c12adf1da61f4d267611948bfdc9293c9b9915da64e41cc1ab68c0167ff1",
                            "8891325197ab8a77689506118fade7bc812944a03bcce47b802e4b8742e590cd",
                            "9fd86e1c5bff348625bab72091d7ea2fa80d197f8e4a5452b378ce8ae0abd9e5"),
    "dirty_carpet": ("6337d3ef1a11ca8bea2d31b86da15a65f10ae92c7f3345fae62924bae194f6c0",
                     "b59b74a4aa2ef9389704019d376cdca1f5f799490632a6aca8583bb3280b9117",
                     "7e06b98f11a304a433916c847be2cbf6308b12a5372fe9961924e70127a833d4"),
    "large_grey_tiles": ("6ee421e93ecf3d68d4788356a9eba76e0aabab620a7055ef57af34da7898850f",
                         "87183caf28504115fa2001be3c7bbfbc2e1da4313a49ddab23ece6cd5286c03e",
                         "a9c2cc6d8607f163797d5a41d9a2304eec18f6d29c9e31ce0c471a3646c11f09"),
    "terrazzo_tiles": ("26cc390c19e468de03fc9699198398590a7adc1ddd4972e6d6fb65d76403bb66",
                       "48f79a943f626c90cfeac4528b765d9f7df862883890a9d05ddf00a2209e0fa4",
                       "23ee634b3550f8e0130406925671a35cc3ebae6be83ee1e543640a19283f64b5"),
}
#: the studio light: Poly Haven ``brown_photostudio_02``, 4k
HDRI = ("brown_photostudio_02", "368251c254a47dc7aaf532a515c0bd9ad3708ce5a3c5bb520b1628bcb9a37bc0")


def looks_dir() -> Path:
    return third_party_dir() / "looks"


def texture_url(name: str, part: str) -> str:
    return f"{POLY}/Textures/jpg/2k/{name}/{name}_{part}_2k.jpg"


def texture(name: str, part: str) -> Path:
    """One map of a texture set (``part`` in MAPS), fetched the first time."""
    if name not in TEXTURES:
        raise ConfigError(f"no texture set {name!r}; have {sorted(TEXTURES)}")
    return _fetched(looks_dir() / "textures" / f"{name}_{part}_2k.jpg", texture_url(name, part),
                    TEXTURES[name][MAPS.index(part)])


def hdri() -> Path:
    name, sha = HDRI
    return _fetched(looks_dir() / "skies" / f"{name}_4k.hdr", f"{POLY}/HDRIs/hdr/4k/{name}_4k.hdr", sha)


def fetch_all() -> list[Path]:
    """Every file, fetched and checked."""
    return [texture(n, p) for n in TEXTURES for p in MAPS] + [hdri()]


def _fetched(path: Path, url: str, sha256: str) -> Path:
    """``path``, downloaded from ``url`` if it is not there yet -- to a temporary name and renamed, so that
    processes starting together never read half a file -- and checked against its pinned SHA-256."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                        timeout=180) as response:
                tmp.write_bytes(response.read())
        except OSError as exc:
            raise ConfigError(f"could not fetch {url} ({exc}); put the file at {path}") from None
        os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ConfigError(f"{path} does not match its pinned SHA-256 ({digest}); delete it to fetch it again")
    return path
