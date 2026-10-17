"""weather.py — Miranda's weather tool: where the board is, and its forecast.

  here()          the board's public IP (api.ipify.org), then its position
                  (ipwho.is; ip-api.com when that fails) — an IP's city is
                  approximate; cached for an hour
  geocode(name)   a named place (Open-Meteo's geocoding API)
  report(place, day)
                  the forecast (Open-Meteo, cached for ten minutes) of "now"
                  (the current conditions), "today", "tomorrow" or a weekday:
                  a Report — sentence() for the chat model, yes_no(question)
                  the answer to a rain / umbrella / snow question

No API keys.  The services see the board's public IP; nothing else is sent.
Units: metric, the words spelled out for text to speech ("degrees Celsius",
"kilometres per hour").
"""

from __future__ import annotations

import datetime
import json
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, Optional

IPIFY = "https://api.ipify.org?format=json"
IPWHO = "https://ipwho.is/{ip}"
IPAPI = "http://ip-api.com/json/{ip}?fields=status,country,regionName,city,lat,lon,timezone"
GEOCODE = "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=1&language=en&format=json"
FORECAST = ("https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
            "&current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
            "wind_speed_10m_max&timezone=auto&forecast_days=7")
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# WMO weather interpretation codes (Open-Meteo)
WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
       51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
       61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "heavy freezing rain",
       71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light rain showers",
       81: "rain showers", 82: "violent rain showers", 85: "light snow showers", 86: "heavy snow showers",
       95: "a thunderstorm", 96: "a thunderstorm with hail", 99: "a thunderstorm with heavy hail"}


class WeatherError(RuntimeError):
    pass


@dataclass(frozen=True)
class Place:
    name: str
    region: str
    country: str
    lat: float
    lon: float
    source: str                      # "ip" (the board's public IP), "named", "configured"

    def label(self) -> str:
        return ", ".join(x for x in (self.name, self.country) if x) or f"{self.lat:.2f}, {self.lon:.2f}"


def http_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "kv260-miranda/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _round(x: float) -> int:
    return int(round(x))


WET = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
SNOW = {71, 73, 75, 77, 85, 86}


def a_an(n: int) -> str:
    """The article before a number said aloud: "an 18", "an 8", "a 97"."""
    return "an" if n in (8, 11, 18) or 80 <= n <= 89 else "a"


def rain_verdict(p: float) -> str:
    """The forecast's chance of rain in words — the chat model repeats a verdict
    better than it judges a number (SmolLM2-360M said "it will likely rain" at 18 %)."""
    return "rain is likely" if p >= 60 else "rain is possible" if p >= 30 else "rain is unlikely"


class Weather:
    def __init__(self, fetch: Callable[[str], dict] = http_json, place_ttl: float = 3600.0,
                 forecast_ttl: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.fetch, self.place_ttl, self.forecast_ttl, self.clock = fetch, place_ttl, forecast_ttl, clock
        self._here: Optional[tuple] = None                 # (time, Place)
        self._forecasts: Dict[tuple, tuple] = {}           # (lat, lon) -> (time, data)
        self._lock = threading.Lock()

    # ── where ──

    def public_ip(self) -> str:
        ip = str(self.fetch(IPIFY).get("ip") or "")
        if not ip:
            raise WeatherError("api.ipify.org gave no IP")
        return ip

    def locate_ip(self, ip: str) -> Place:
        try:
            d = self.fetch(IPWHO.format(ip=ip))
            if d.get("success"):
                return Place(d.get("city") or "", d.get("region") or "", d.get("country") or "",
                             float(d["latitude"]), float(d["longitude"]), "ip")
        except Exception:                                             # noqa: BLE001 — the second service
            pass
        d = self.fetch(IPAPI.format(ip=ip))
        if d.get("status") != "success":
            raise WeatherError(f"no position for the IP {ip}")
        return Place(d.get("city") or "", d.get("regionName") or "", d.get("country") or "",
                     float(d["lat"]), float(d["lon"]), "ip")

    def here(self) -> Place:
        """The board's position from its public IP (cached)."""
        with self._lock:
            if self._here and self.clock() - self._here[0] < self.place_ttl:
                return self._here[1]
        place = self.locate_ip(self.public_ip())
        with self._lock:
            self._here = (self.clock(), place)
        return place

    def geocode(self, name: str) -> Optional[Place]:
        d = self.fetch(GEOCODE.format(name=urllib.parse.quote(name)))
        res = d.get("results") or []
        if not res:
            return None
        r = res[0]
        return Place(r.get("name") or name, r.get("admin1") or "", r.get("country") or "",
                     float(r["latitude"]), float(r["longitude"]), "named")

    # ── the forecast ──

    def forecast(self, place: Place) -> dict:
        key = (round(place.lat, 2), round(place.lon, 2))
        with self._lock:
            hit = self._forecasts.get(key)
            if hit and self.clock() - hit[0] < self.forecast_ttl:
                return hit[1]
        d = self.fetch(FORECAST.format(lat=place.lat, lon=place.lon))
        if "current" not in d or "daily" not in d:
            raise WeatherError("api.open-meteo.com gave no forecast")
        with self._lock:
            self._forecasts[key] = (self.clock(), d)
        return d

    def report(self, place: Place, day: str = "now") -> "Report":
        """The forecast of a day ("now": the current conditions)."""
        d = self.forecast(place)
        if day == "now":
            c = d["current"]
            code = int(c["weather_code"])
            return Report(place, "now", "", WMO.get(code, "unknown conditions"), code,
                          temp=_round(c["temperature_2m"]), feels=_round(c["apparent_temperature"]),
                          wind=_round(c["wind_speed_10m"]), humidity=_round(c["relative_humidity_2m"]))
        daily = d["daily"]
        dates = [datetime.date.fromisoformat(s) for s in daily["time"]]
        i = day_index(day, dates[0])
        if i is None or i >= len(dates):
            raise WeatherError(f"no forecast for {day}")
        code = int(daily["weather_code"][i])
        return Report(place, "today" if i == 0 else "tomorrow" if i == 1 else dates[i].strftime("%A"),
                      dates[i].strftime("%A %d %B"), WMO.get(code, "unknown conditions"), code,
                      tmin=_round(daily["temperature_2m_min"][i]), tmax=_round(daily["temperature_2m_max"][i]),
                      rain=_round(daily["precipitation_probability_max"][i] or 0),
                      wind=_round(daily["wind_speed_10m_max"][i]))


@dataclass
class Report:
    place: Place
    when: str                        # "now", "today", "tomorrow", a weekday
    date: str                        # "Sunday 18 October" ("" now)
    conditions: str                  # WMO's words
    code: int
    temp: Optional[int] = None       # now
    feels: Optional[int] = None
    humidity: Optional[int] = None
    tmin: Optional[int] = None       # a day
    tmax: Optional[int] = None
    rain: Optional[int] = None       # the day's chance of rain, percent
    wind: int = 0                    # now / the day's highest, km/h

    @property
    def state(self) -> str:
        """Now: "snowing", "raining" or "dry"."""
        return "snowing" if self.code in SNOW else "raining" if self.code in WET else "dry"

    def sentence(self) -> str:
        """The report as one sentence for the chat model."""
        where = self.place.label()
        if self.when == "now":
            wet = {"snowing": "it is snowing", "raining": "it is raining"}.get(self.state, "it is not raining")
            return (f"Weather in {where} now: {self.conditions} ({wet}), {self.temp} degrees Celsius (feels like "
                    f"{self.feels}), wind {self.wind} kilometres per hour, humidity {self.humidity} percent.")
        return (f"Forecast for {where} {self.when} ({self.date}): {self.conditions}, {self.tmin} to {self.tmax} "
                f"degrees Celsius, chance of rain {self.rain} percent ({rain_verdict(self.rain)}), wind up to "
                f"{self.wind} kilometres per hour.")

    def yes_no(self, question: str) -> str:
        """The answer to a rain / umbrella / snow question, from the numbers (the chat
        model says yes to these whatever the data)."""
        q = question.lower()
        where = self.place.label()
        if "snow" in q:
            snow = self.code in SNOW
            lead = "Yes, snow is forecast" if snow else "No, no snow is forecast"
        elif self.when == "now":
            lead = {"raining": "Yes, it is raining", "snowing": "It is snowing"}.get(self.state, "No, it is not raining")
        elif "umbrella" in q or "raincoat" in q:
            lead = ("Yes, take an umbrella" if self.rain >= 60 else "Maybe take an umbrella" if self.rain >= 30
                    else "You probably won't need an umbrella")
        else:
            lead = ("Yes, rain is likely" if self.rain >= 60 else "Maybe, rain is possible" if self.rain >= 30
                    else "Probably not, rain is unlikely")
        if self.when == "now":
            return f"{lead} in {where} now: {self.conditions}, {self.temp} degrees Celsius."
        when = self.when if self.when in ("today", "tomorrow") else f"on {self.when}"
        return (f"{lead} {when} in {where}: {a_an(self.rain)} {self.rain} percent chance of rain, {self.conditions}, "
                f"{self.tmin} to {self.tmax} degrees Celsius.")


def day_index(day: str, first: datetime.date) -> Optional[int]:
    """The daily forecast's index of a day word, the first day being the place's today."""
    day = day.lower()
    if day == "today":
        return 0
    if day == "tomorrow":
        return 1
    if day in WEEKDAYS:
        return (WEEKDAYS.index(day) - first.weekday()) % 7
    return None
