"""Abstract base class for event stream generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import pynapple as nap


class EventGenerator(ABC):
    """Pure-interface base class for all event stream generators.

    Subclasses must implement :meth:`generate`.  This class provides the
    shared IO methods :meth:`generate_to_file` and :meth:`load_from_file`.
    """

    @abstractmethod
    def generate(self, labels: Optional[Sequence[str]] = None) -> nap.TsGroup:
        """Generate event times for all streams.

        Args:
            labels: Optional override for stream labels.

        Returns:
            TsGroup mapping stream indices to Ts objects (timestamps in ms).
        """
        pass

    def generate_to_file(
        self, path: str, labels: Optional[Sequence[str]] = None
    ) -> nap.TsGroup:
        """Generate events and save to file with times in ms."""
        tsgroup = self.generate(labels=labels)
        with open(path, "w", encoding="utf-8") as f:
            for idx, ts in tsgroup.items():
                lab = (
                    tsgroup.get_info("label")[idx]
                    if "label" in tsgroup.metadata
                    else str(idx)
                )
                # Pynapple stores times in seconds; convert to ms for file output
                times_ms = ts.as_units("ms").index.values
                if len(times_ms) > 0:
                    f.write(lab + " " + " ".join(str(t) for t in times_ms) + "\n")
                else:
                    f.write(lab + "\n")
        return tsgroup

    @staticmethod
    def load_from_file(path: str) -> nap.TsGroup:
        """Load event times from file as TsGroup."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        ts_dict = {}
        labels = []
        all_times = []
        idx = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label = parts[0]
            labels.append(label)
            if len(parts) == 1:
                ts_dict[idx] = nap.Ts(t=[], time_units="ms")
            else:
                times = [float(t) for t in parts[1:]]
                all_times.extend(times)
                ts_dict[idx] = nap.Ts(t=times, time_units="ms")
            idx += 1

        # Create time support from data range or default
        if all_times:
            max_t = max(all_times)
            time_support = nap.IntervalSet(
                start=[0], end=[max(max_t, 1.0)], time_units="ms"
            )
        else:
            time_support = nap.IntervalSet(start=[0], end=[1.0], time_units="ms")

        return nap.TsGroup(ts_dict, label=labels, time_support=time_support)
