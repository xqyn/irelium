'''
irelium
XQ
2026-05-20
Loss history tracker with EMA smoothing.
'''


class LossHistory:
    '''
    Tracks raw and smoothed loss via exponential moving average.

    Args:
        smoothing_factor: EMA alpha (0 = no smoothing, 0.99 = heavy smoothing).
    '''

    def __init__(self, smoothing_factor: float = 0.99) -> None:
        self.smoothing_factor = smoothing_factor
        self.raw    = []
        self.smooth = []
        self._ema   = None

    def append(self, loss: float) -> None:
        '''Append a new loss value and update EMA.'''
        self.raw.append(loss)
        self._ema = loss if self._ema is None else (
            self.smoothing_factor * self._ema
            + (1 - self.smoothing_factor) * loss
        )
        self.smooth.append(self._ema)

    def reset(self) -> None:
        '''Clear all history and reset EMA.'''
        self.raw    = []
        self.smooth = []
        self._ema   = None

    def __len__(self) -> int:
        return len(self.raw)