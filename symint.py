import numpy as np
import scipy.stats


def is_symmetric(errs, critical_p: float = 0.01):
    """
    Is distribution symmetric.
    """
    return scipy.stats.wilcoxon(errs).pvalue <= critical_p


def round_dict_values(func):
    def rounder(*args, **kwargs):
        out = func(*args, **kwargs)
        if not isinstance(out, dict):
            raise ValueError('output is expected to be dict')
        return {k: round(v, 4) for k, v in out.items()}

    return rounder


def symetric_percentile(arr, percentile):
    """
    Find central percentiles of the symmetric
    distribution.
    """
    return np.percentile(np.sort(np.abs(arr)), percentile)


class AsymmetricDistException(Exception):
    def __init__(self, message="the distribution is asymmetric!"):
        self.message = message
        super().__init__(self.message)


def test_asymmetricity():
    """
    Check asymmetricity test.
    """
    arr = np.random.normal(loc=0, scale=1, size=300)
    intopt = OptErrorIntervals(arr, k=10, uncertainty=0.05)
    try:
        intopt.locate()
        return False
    except AsymmetricDistException:
        return True
    except:
        return False


# assert test_asymmetricity()


class OptErrorIntervals:

    def __init__(self, arr: np.array, k: int, uncertainty=0.05):
        self.k = k
        self.ci = (1 - uncertainty) * 100
        self.arr = arr
        self.symmetric = is_symmetric(self.arr)

    @property
    def unit_scale(self):
        return 1 / (self.k - 1)

    @property
    def deviation(self):
        return symetric_percentile(self.arr, self.ci)

    def locate(self):
        if self.symmetric:
            point = self.deviation
            return -point, +point
        else:
            raise AsymmetricDistException()

    @round_dict_values
    def evaluate(self):
        return {
            'k': self.k,
            'unit_scale': self.unit_scale,
            'ci': self.ci,
            'num_sizes_covers': (coverage := (self.deviation // self.unit_scale) * 2 + 1),
            'sensitivity': 1 - (coverage / self.k),
        }