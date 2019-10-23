#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:42:07 2018

@author: allen
"""
import datetime
import pandas as pd
from functools import lru_cache

def _to_hours_mins_secs(time_taken):
    """Convert seconds to hours, mins, and seconds."""
    mins, secs = divmod(time_taken, 60)
    hours, mins = divmod(mins, 60)
    return hours, mins, secs

def convert_str_to_timestamp(str_dt):
    return pd.Timestamp(str_dt)

def convert_timestamp_to_str(timestamp, format='%Y-%m-%d'):
    return convert_timestamp_to_dt(timestamp).strftime(format)

def convert_timestamp_to_dt(timestamp):
    return timestamp.to_pydatetime()


def convert_str_to_dt(str_dt, format_="%Y-%m-%d %H:%M:%S"):
    """convert str tpye to datetime"""
    # "%Y-%m-%d %H:%M:%S.%f"
    # "%m/%d/%Y %H:%M:%S.%f"
    dt = datetime.datetime.strptime(str_dt, format_)
    return dt

def convert_date_to_date_int(dt):
    t = dt.year * 10000 + dt.month * 100 + dt.day
    return t


def convert_date_to_int(dt):
    t = dt.year * 10000 + dt.month * 100 + dt.day
    t *= 1000000
    return t


def convert_dt_to_int(dt):
    t = convert_date_to_int(dt)
    t += dt.hour * 10000 + dt.minute * 100 + dt.second
    return t


def convert_int_to_date(dt_int):
    dt_int = int(dt_int)
    if dt_int > 100000000:
        dt_int //= 1000000
    return _convert_int_to_date(dt_int)


@lru_cache(None)
def _convert_int_to_date(dt_int):
    year, r = divmod(dt_int, 10000)
    month, day = divmod(r, 100)
    return datetime.datetime(year, month, day)


@lru_cache(20480)
def convert_int_to_datetime(dt_int):
    dt_int = int(dt_int)
    year, r = divmod(dt_int, 10000000000)
    month, r = divmod(r, 100000000)
    day, r = divmod(r, 1000000)
    hour, r = divmod(r, 10000)
    minute, second = divmod(r, 100)
    return datetime.datetime(year, month, day, hour, minute, second)


def convert_ms_int_to_datetime(ms_dt_int):
    dt_int, ms_int = divmod(ms_dt_int, 1000)
    dt = convert_int_to_datetime(dt_int).replace(microsecond=ms_int * 1000)
    return dt


def convert_date_time_ms_int_to_datetime(date_int, time_int):
    date_int, time_int = int(date_int), int(time_int)
    dt = _convert_int_to_date(date_int)

    hours, r = divmod(time_int, 10000000)
    minutes, r = divmod(r, 100000)
    seconds, millisecond = divmod(r, 1000)

    return dt.replace(hour=hours, minute=minutes, second=seconds,
                      microsecond=millisecond * 1000)