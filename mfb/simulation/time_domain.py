from __future__ import annotations

from dataclasses import dataclass

from mfb.config.params import MFBConfiguration
from mfb.control.limiter import LimiterParams, SignalLimiter
from mfb.control.loop_assembler import MFBLoop
from mfb.control.prefilter import PrefilterModel, PrefilterParams
from mfb.plant.source import SourceModel
from mfb.plant.summing_amp import SummingAmplifierModel, SummingAmplifierParams
from mfb.simulation.signal_flow import ClosedLoopSignalFlow, TimeDomainTrace


@dataclass(frozen=True, slots=True)
class TimeDomainExperimentConfig:
    prefilter: PrefilterParams = PrefilterParams()
    limiter: LimiterParams = LimiterParams()
    summer: SummingAmplifierParams = SummingAmplifierParams()


class TimeDomainExperimentRunner:
    def __init__(self, config: MFBConfiguration, experiment: TimeDomainExperimentConfig | None = None) -> None:
        self.config = config
        self.experiment = TimeDomainExperimentConfig() if experiment is None else experiment

    def run(self) -> TimeDomainTrace:
        loop = MFBLoop(self.config)
        source = SourceModel(self.config.source, self.config.signal)
        flow = ClosedLoopSignalFlow(
            loop=loop,
            source=source,
            prefilter=PrefilterModel(self.experiment.prefilter, self.config.signal.sample_rate_hz),
            limiter=SignalLimiter(self.experiment.limiter),
            summer=SummingAmplifierModel(self.experiment.summer),
        )
        return flow.run()
