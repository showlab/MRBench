"""One physical playback path for scripted references."""


def play_script(inst, script, settle_s=.5, video=True, *, view=None, fps=20.,
                device=None, progress=False):
    """Load, optionally warm the cameras, and play the complete checked script.

    The caller launches Isaac and owns scoring, report format and video encoding.
    Returning the frames first preserves diagnostics even if encoding fails.
    """
    from .engines import make_engine
    from .engines.isaac.puppeteer import Puppeteer
    from .policies.script import phase
    from .render import SIZE, default_views

    options = dict(headless=True,
                   cameras={'main': view or default_views()['main']} if video else {},
                   camera_size=SIZE, wrist_cameras=video)
    if device is not None:
        options['device'] = device
    engine = make_engine('isaac', **options)
    if progress:
        print('Loading scene', flush=True)
    engine.load(inst.scene, inst.seed)
    if video:
        engine.render()

    class ReferencePuppeteer(Puppeteer):
        last_note = None

        def _step(self, plan, t_next):
            if progress and self.note and self.note != self.last_note:
                print(f'[{self.t:5.1f}s] {self.note}', flush=True)
                self.last_note = self.note
            return super()._step(plan, t_next)

    pup = ReferencePuppeteer(engine, inst, camera='main' if video else None, fps=fps)
    if progress:
        print('Planning and playing reference', flush=True)
    frames = pup.play([phase(settle_s, '')] + script)
    return pup, frames
