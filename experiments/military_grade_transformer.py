# stolen from https://github.com/sohandillikar/SupportResistance/blob/main/support_resistance.py

# Standard Library
from math import sqrt

# Third party dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

# Current project dependencies
import file_combiner as fc


def pythag(pt1, pt2):
    a_sq = (pt2[0] - pt1[0]) ** 2
    b_sq = (pt2[1] - pt1[1]) ** 2
    return sqrt(a_sq + b_sq)


def regression_ceof(pts):
    X = np.array([pt[0] for pt in pts]).reshape(-1, 1)
    y = np.array([pt[1] for pt in pts])
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_[0], model.intercept_


def local_min_max(pts):
    local_min = []
    local_max = []
    prev_pts = [(0, pts[0]), (1, pts[1])]
    for i in range(1, len(pts) - 1):
        append_to = ""
        if pts[i - 1] > pts[i] < pts[i + 1]:
            append_to = "min"
        elif pts[i - 1] < pts[i] > pts[i + 1]:
            append_to = "max"
        if append_to:
            if local_min or local_max:
                prev_distance = pythag(prev_pts[0], prev_pts[1]) * 0.5
                curr_distance = pythag(prev_pts[1], (i, pts[i]))
                if curr_distance >= prev_distance:
                    prev_pts[0] = prev_pts[1]
                    prev_pts[1] = (i, pts[i])
                    if append_to == "min":
                        local_min.append((i, pts[i]))
                    else:
                        local_max.append((i, pts[i]))
            else:
                prev_pts[0] = prev_pts[1]
                prev_pts[1] = (i, pts[i])
                if append_to == "min":
                    local_min.append((i, pts[i]))
                else:
                    local_max.append((i, pts[i]))
    return local_min, local_max


ticker = "TSLA"
interval = "1min"
df = pd.read_parquet(fc.get_ticker_file_parquet(ticker, interval), engine="fastparquet")

series = df[::-1].tail(100)["close"]
series.index = np.arange(series.shape[0])

month_diff = series.shape[0] // 30
if month_diff == 0:
    month_diff = 1

smooth = int(2 * month_diff + 3)

pts = savgol_filter(series, smooth, 3)

local_min, local_max = local_min_max(pts)

local_min_slope, local_min_int = regression_ceof(local_min)
local_max_slope, local_max_int = regression_ceof(local_max)
support = (local_min_slope * np.array(series.index)) + local_min_int
resistance = (local_max_slope * np.array(series.index)) + local_max_int

plt.title(ticker)
plt.xlabel("Bars")
plt.ylabel("Prices")
plt.plot(series, label=ticker)
plt.plot(pts, label="smooth", c="o")
plt.plot(support, label="Support", c="r")
plt.plot(resistance, label="Resistance", c="g")
plt.legend()
plt.savefig(f"{ticker}_support_resistance_{interval}.png")
plt.show()
