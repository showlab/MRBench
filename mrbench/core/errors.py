"""Exception types shared across the package."""


class MRBenchError(Exception):
    """Base class for every error MultiRobotBench raises on purpose."""


class ConfigError(MRBenchError):
    """A task card, robot model or run configuration is invalid."""


class EngineNotAvailable(MRBenchError):
    """The requested engine cannot run in this interpreter (e.g. no Isaac Sim)."""


class AssetMissing(MRBenchError):
    """A robot description or USD has not been fetched / converted yet."""
