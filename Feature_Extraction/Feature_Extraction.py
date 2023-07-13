import numpy as np
import bottleneck as bn
import pandas as pd
import warnings
from functools import wraps
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


def sliding_windows(data, window_size, window_step):
    sliding_windows = []
    for win_step, win_size, data in zip(window_size, window_step, data):
        d = []
        for step in range(0, len(data) // win_size * win_size, win_step):
            d.append(data[step:step + win_size])
        sliding_windows.append(np.array(d))
    return sliding_windows


def generate_sliding_windows(only_targets=False):
    def gg_sliding_windows(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.sliding_windows is None:
                if not only_targets:
                    sliding_win = sliding_windows(self.data, self.window_size, self.window_step)
                    self.sliding_windows = sliding_win
                else:
                    try:
                        t_sliding_win = sliding_windows([self.targets], self.t_window_size, self.t_window_step)
                        self.t_sliding_windows = t_sliding_win
                    except AttributeError:
                        # targets not defined
                        pass
            return func(self, *args, **kwargs)
        return wrapper
    return gg_sliding_windows


def count_calls(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if func.__name__ not in self.funcs_called:
            self.funcs_called[func.__name__] = 1
        else:
            self.funcs_called[func.__name__] += 1
        if self.funcs_called[func.__name__] > 1:
            warnings.warn("Function {} more than once called!".format(func.__name__))
        return func(self, *args, **kwargs)
    return wrapper


class FeatureExtraction:
    def __init__(self, *data, window_length, sample_rate, window_step=None, **kwargs):
        """
        Calls a sliding windows on a bunch of data and does the specified functions
        data: iterables in which the data is stored
        targets: iterable in which the targets are stored
        window_length: length in seconds for each window
        sample_rate: in Hz, needed for each data given
        window_step: step in seconds of each window, can also be specified via window overlap or percentage overlap

        Everything in time domain -- frequency domain needed?

        sources: https://ieeexplore.ieee.org/document/7748960
        https://osf.io/rd2cf/download
        https://www.mdpi.com/1424-8220/22/16/5948

        todo: medium differential value;
        todo: amplitude first burst
        todo: maximal fractal length
        todo: mean absolute deviation
        todo: difference mean absolute value

        todo: frequency domain features ??
        """
        assert len(data) == len(sample_rate)
        self.sample_rate = sample_rate
        self.window_step = [int(window_step * s_rate) if window_step is not None else int(window_length * s_rate)
                            for s_rate in self.sample_rate]
        self.window_size = [int(window_length * s_rate) for s_rate in self.sample_rate]
        self.data = data
        self.extracted_feat_data = {}
        self.sliding_windows = None
        self.t_sliding_windows = None
        self.funcs_called = {}
        self.t_mode = None
        if all([kwargs.get("targets") is not None, kwargs.get("t_sample_rate") is not None]):
            self.t_window_step = [int(window_step * s_rate) if window_step is not None else int(window_length * s_rate)
                                  for s_rate in kwargs.get("t_sample_rate")]

            self.t_window_size = [int(window_length * s_rate) for s_rate in kwargs.get("t_sample_rate")]
            self.targets = kwargs.get("targets")
            self.t_mode = kwargs.get("t_mode")

    @classmethod
    def from_window_overlap(cls, *data, targets, window_size, sample_rate, overlap):
        if overlap > window_size:
            raise ValueError("overlap cannot be larger than window_size")
        return cls(*data, targets=targets,
                   window_length=window_size,
                   sample_rate=sample_rate,
                   window_step=window_size - overlap)

    @classmethod
    def from_percent_overlap(cls, *data, targets, window_size, sample_rate, overlap_percent):
        if not 0 <= overlap_percent <= 1:
            raise ValueError("Please pass a number between 0 and 1 for overlap_percent")
        return cls(data, targets=targets,
                   window_length=window_size,
                   sample_rate=sample_rate,
                   window_step=window_size - int(overlap_percent * window_size))

    @count_calls
    def mean(self):
        """
        calculates the mean on each window
        """
        self.extracted_feat_data["mean"] = []
        for win_step, win_size, data in zip(self.window_step, self.window_size, self.data):
            self.extracted_feat_data["mean"].append(list(bn.move_mean(np.abs(data),
                                                                      win_size)[win_size - 1::win_step]))
        return self

    @count_calls
    def sta_dev(self):
        """
        calculates the standard deviation on each window
        """
        self.extracted_feat_data["STD"] = []
        for win_step, win_size, data in zip(self.window_step, self.window_size, self.data):
            self.extracted_feat_data["STD"].append(list(bn.move_std(data,
                                                                    win_size)[win_size - 1::win_step]))
        return self

    @count_calls
    def min(self):
        """
        calculates the min of each window
        """
        self.extracted_feat_data["min"] = []
        for win_step, win_size, data in zip(self.window_step, self.window_size, self.data):
            self.extracted_feat_data["min"].append(list(bn.move_min(data,
                                                                    win_size)[win_size - 1::win_step]))
        return self

    @count_calls
    def max(self):
        """
        calculates the max of each window
        -- same as amplitude of the first burst ?
        """
        self.extracted_feat_data["max"] = []
        for win_step, win_size, data in zip(self.window_step, self.window_size, self.data):
            self.extracted_feat_data["max"].append(list(bn.move_max(data,
                                                                    win_size)[win_size - 1::win_step]))
        return self

    def amplitude_first_burst(self):
        """
        Haven't really understood what this is, but: the maximum value of the amplitude contains the amplitude of the
        first burst (Electromyography parameter variations with electrocardiography Noise - Chang, Lui et al.

        """
        pass

    @count_calls
    def variance(self):
        """
        calculates the variance of each window
        """
        self.extracted_feat_data["VAR"] = []
        for win_step, win_size, data in zip(self.window_step, self.window_size, self.data):
            self.extracted_feat_data["VAR"].append(list(bn.move_var(data,
                                                                    win_size)[win_size - 1::win_step]))
        return self

    @count_calls
    @generate_sliding_windows()
    def zero_crossing(self, threshold):
        """
        calculates the zero crossing on each window
        """
        self.extracted_feat_data["ZC"] = []
        for ary in self.sliding_windows:
            zero_cross = ary[:, 1:] * ary[:, :-1] <= 0
            thres = np.abs(ary[:, :-1] - ary[:, 1:]) >= threshold
            self.extracted_feat_data["ZC"].append(np.count_nonzero(np.all(np.dstack((zero_cross, thres)), axis=2),
                                                                   axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def willison_amp(self, threshold):
        """
        calculates the willision amplitude (counts the occurence when amplitude exceeds threshold) is on each window
        """
        self.extracted_feat_data["will_amp"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["will_amp"].append(np.count_nonzero(
                (np.abs(ary[:, 1:] - ary[:, :-1]) > threshold), axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def average_amplitude_change(self):
        """
        calculates the average amplitude change on each window
        """
        self.extracted_feat_data["AAC"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["AAC"].append(np.mean(np.abs(ary[:, 1:] - ary[:, :-1]), axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def mean_absolut_value(self, type_=0, **kwargs):
        """
        calculates the mean absolut value one each window. If wanted keyword arguments can be specified to weight
        the samples depending on where in the sequence they stand
        -- usefunc: specify if weighted func should be used - standard func will be used
        -- func: define a func which returns a weighted array
           --> also called MAV_2 set type_ to 2
        -- bound: boundaries in which a weight is applied
        -- val: weighted value within the bound
        -- other: weighted value outside the boundaries
           --> also called MAV_1 - set type_ to 1
        ## weighting probably only useful when EMG-bursts are somehow detected
        """
        if type_ == 1:
            kwargs = {"bounds": [0.25, 0.75], "val": 1, "other": 0.5}
        elif type_ == 2:
            kwargs = {"usefunc": True}
        self.extracted_feat_data["MAV"] = []
        for ary in self.sliding_windows:
            weight = np.ones_like(ary)
            if all([kwargs.get("bounds"), kwargs.get("val"), kwargs.get("other")]):
                kwargs["val"] = kwargs["val"] if isinstance(kwargs["val"], list) else [kwargs["val"]]
                kwargs["bounds"] = kwargs["bounds"] if isinstance(kwargs["bounds"][0], list) else [kwargs["bounds"]]
                assert len(kwargs["bounds"]) == len(kwargs["val"])
                weight = weight * kwargs["other"]
                for bound, value in zip(kwargs["bounds"], kwargs["val"]):
                    weight[:, int((ln := ary.shape[1]) * bound[0]):int(ln * bound[1])] = value
            elif kwargs.get("func"):
                weight = kwargs["func"]()
            elif kwargs.get("usefunc"):
                weight = self.weighted_func(*ary.shape)
            print(weight)
            self.extracted_feat_data["MAV"].append(np.mean(weight * np.abs(ary), axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def root_mean_square(self):
        self.extracted_feat_data["RMS"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["RMS"].append(np.sqrt(np.mean(np.square(ary), axis=1)))
        return self

    @count_calls
    @generate_sliding_windows()
    def simple_square_integral(self):
        self.extracted_feat_data["SSI"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["SSI"].append(np.square(ary), axis=1)
        return self

    @generate_sliding_windows()
    @count_calls
    def integrated_absolut_value(self):
        """
        also called integrated EMG
        """
        self.extracted_feat_data["IAV"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["IAV"].append(np.abs(ary), axis=1)
        return self

    @generate_sliding_windows()
    @count_calls
    def v_order(self, order):
        self.extracted_feat_data["V"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["V"].append((1 / ary.shape[1] * np.sum(ary ** order, axis=1)) ** (1 / order))
        return self

    @generate_sliding_windows()
    @count_calls
    def waveform_length(self):
        self.extracted_feat_data["WL"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["WL"].append(np.sum(np.abs(ary[:, 1:] - ary[:, :-1]), axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def difference_absolute_std_value(self):
        self.extracted_feat_data["DASDV"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["DASDV"].append(
                (ary.shape[1] - 1) * np.sum((ary[:, 1:] - ary[:, :-1]) ** 2, axis=1))
        return self

    @generate_sliding_windows()
    @count_calls
    def slope_sign_change(self, threshold=1):
        self.extracted_feat_data["SSC"] = []
        for ary in self.sliding_windows:
            x_iplus1 = ary[:, 2:]
            x_i = ary[:, 1:-1]
            x_iminus1 = ary[:, :-2]

            a = np.all(np.dstack((x_iminus1 < x_i, x_iplus1 < x_i)), axis=2)
            b = np.all(np.dstack((x_iminus1 > x_i, x_iplus1 > x_i)), axis=2)

            th = np.any(np.dstack((np.abs(x_i - x_iplus1) >= threshold, np.abs(x_i - x_iminus1) >= threshold)), axis=2)

            mask = np.all(np.dstack((th, np.any(np.dstack((a, b)), axis=2))), axis=2)
            self.extracted_feat_data["SSC"].append(np.count_nonzero(mask, axis=1))
        return self

    @count_calls
    @generate_sliding_windows()
    def myopulse_percentage_rate(self, threshold):
        self.extracted_feat_data["MYOP"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["MYOP"].append(1 / ary.shape[1] * np.count_nonzero(np.abs(ary) > threshold),
                                                    axis=1)
        return self

    @count_calls
    @generate_sliding_windows()
    def temporal_moments(self, order):
        if order in [3, 4, 5]:
            self.extracted_feat_data["TM" + str(order)] = []
            for ary in self.sliding_windows:
                self.extracted_feat_data["TM" + str(order)].append(1 / ary.shape[1] * np.sum(ary ** order, axis=1))
        return self

    @count_calls
    @generate_sliding_windows()
    def log_difference_absolute_mean_value(self):
        self.extracted_feat_data["LDAMV"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["LDAMV"].append(np.log(1 / ary.shape[1] *
                                                            np.sum(np.abs(ary[:, 1:] - ary[:, :-1]), axis=1)))
        return self

    @count_calls
    @generate_sliding_windows()
    def log_detector(self):
        self.extracted_feat_data["LOG"] = []
        for ary in self.sliding_windows:
            self.extracted_feat_data["LOG"].append(1 / ary.shape[1] * np.sum(np.log(np.np.abs(ary)), axis=1))
        return self

    @generate_sliding_windows(True)
    def adapt_targets(self):
        """
        adapts targets to the sliding windows.
        :return: 1d-target-array
        """
        if self.t_mode is None:
            return
        if self.t_mode.upper() == "AVERAGE":
            self.extracted_feat_data["target"] = np.mean(self.t_sliding_windows, axis=2)
        elif self.t_mode.upper() == "MOST FREQUENT":
            self.extracted_feat_data["targets"] = np.argmax(np.apply_along_axis(np.bincount,
                                                                                2, self.t_sliding_windows,
                                                                                None,
                                                                                np.max(self.t_sliding_windows) + 1),
                                                            axis=2)

    def __call__(self, *args, **kwargs):
        """
        Return everything calculated
        :param args:
        :param kwargs:
        :return:
        """
        self.adapt_targets()
        if kwargs["type"].upper() == "PANDAS":
            return pd.DataFrame({name + str(i): val[i]
                                 for name, val in self.extracted_feat_data.items() for i in range(len(val))})
        elif kwargs["type"].upper() == "NUMPY":
            return np.array([v for (key, val)
                             in self.extracted_feat_data.items() for v in val if key != "targets"]).T, \
                np.array(self.extracted_feat_data.get("targets")).T

    @staticmethod
    def weighted_func(self, height, width):
        w = []
        for i in range(width):
            if 0.25 * width <= i <= 0.75 * width:
                w.append(1)
            elif i < 0.25 * width:
                w.append(4 * i / width)
            else:
                # from formula, it should be i - width but gets negative???
                w.append(4 * (i - width) / width)
        return np.repeat(np.array(w)[np.newaxis, :], height, axis=0)


class DigitalFilters:
    def __init__(self, *data, window_length, sample_rate, window_step=None, **kwargs):
        self.data = data
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.window_step = [int(window_step * s_rate) if window_step is not None else int(window_length * s_rate)
                            for s_rate in self.sample_rate]
        self.window_size = [int(window_length * s_rate) for s_rate in self.sample_rate]
        self.extracted_feat_data = {}
        self.sliding_windows = None

    @generate_sliding_windows()
    def rms_filter(self, filter_period):
        for chan_data in self.sliding_windows:
            plt.plot(chan_data[0, :])
            plt.show()

    def butterworth(self):
        pass

    @generate_sliding_windows()
    def moving_average(self, filter_period, norm=None):
        """
        Downsampling is needed, to have same length of all the channels
        calculates the moving average filter over a defined filter period
            -- normalize (between 0 and 1) within each window or within each channel-window -
        :return:
        """
        filter_data = []
        for chan_data in self.sliding_windows:
            filter_data.append(uniform_filter1d(np.abs(np.vstack(chan_data)), filter_period, mode="constant", axis=1))
        filter_data = np.dstack(filter_data)
        if norm is not None:
            if norm == "channels":
                min_ = np.tile(np.min(filter_data, axis=(1, 2))[:, np.newaxis], (1, filter_data.shape[2]))
                max_ = np.tile(np.max(filter_data, axis=(1, 2))[:, np.newaxis], (1, filter_data.shape[2]))
            elif norm == "window":
                min_, max_ = np.min(filter_data, axis=1), np.max(filter_data, axis=1)
            else:
                min_, max_ = 0, 1 # Nothing happens
            filter_data = (filter_data.transpose((1, 0, 2)) - min_) / (max_ - min_)
            filter_data = filter_data.transpose((1, 0, 2))
        return filter_data

# Teager–Kaiser energy operator


if __name__ == "__main__":
    np.random.seed(5)
    switch = True
    if switch:
        # Two different ways to do feature extraction, either:
        m = FeatureExtraction(np.arange(140), np.arange(300), np.arange(75),
                              window_length=0.2, sample_rate=[100, 200, 50])
        df = m.zero_crossing(0).mean().max()(type="numpy")
        print(df)
    else:
        m = DigitalFilters(np.random.random(1000) - 0.5, np.random.random(1000) - 1,
                           window_length=0.2, sample_rate=[500, 500])
        m.moving_average(20)

