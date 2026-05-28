class LossHistory:
    '''
    Tracks raw and smoothed loss via exponential moving average.

    Args:
        smoothing_factor: EMA smoothing (0=no smooth, 0.99=heavy smooth).
    '''
    def __init__(self, smoothing_factor: float = 0.99):
        self.smoothing_factor = smoothing_factor
        self.raw    = []
        self.smooth = []
        self._ema   = None

    def append(self, loss: float):
        self.raw.append(loss)
        if self._ema is None:
            self._ema = loss                                      # first value
        else:
            self._ema = self.smoothing_factor * self._ema \
                      + (1 - self.smoothing_factor) * loss        # EMA update
        self.smooth.append(self._ema)

    def get(self) -> list:
        return self.smooth
    