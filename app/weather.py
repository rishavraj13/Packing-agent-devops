"""Open-Meteo client. No API key, no signup.

Two HTTP calls per lookup (geocode the city, then fetch the forecast), hidden
behind one function so the model sees a single tool.

Nothing here raises on failure. Every problem comes back as a WeatherError
carrying a sentence the model can read and act on — a crash mid-loop would
kill the turn, whereas a readable failure lets the agent tell the user the
lookup failed instead of quietly packing from priors.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx

from .models import DayForecast, Forecast

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo's free forecast horizon.
MAX_FORECAST_DAYS = 16

# WMO 4677 weather codes, collapsed to plain descriptions.
WMO: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherError(Exception):
    """Carries a message written for the model, not for a stack trace."""


def _get(client: httpx.Client, url: str, params: dict) -> dict:
    try:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except httpx.TimeoutException as exc:
        raise WeatherError(
            f"Weather service timed out ({url.rsplit('/', 1)[-1]}). "
            "Tell the user the lookup failed and ask whether to retry."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise WeatherError(
            f"Weather service returned HTTP {exc.response.status_code}. "
            "Tell the user the lookup failed and ask whether to retry."
        ) from exc
    except httpx.HTTPError as exc:
        raise WeatherError(
            f"Could not reach the weather service: {exc}. "
            "Tell the user the lookup failed; do not invent a forecast."
        ) from exc
    except ValueError as exc:
        raise WeatherError("Weather service returned malformed JSON.") from exc


def geocode(city: str, client: httpx.Client) -> list[dict]:
    payload = _get(
        client,
        GEOCODE_URL,
        {"name": city, "count": 5, "language": "en", "format": "json"},
    )
    return payload.get("results") or []


def _clamp_window(start_date: str, days: int) -> tuple[str, str, str]:
    """Fit the requested window inside Open-Meteo's horizon.

    Returns (start, end, note). Beyond 16 days there is no forecast to be had,
    and inventing one would defeat the entire point of the project — so the
    note says so plainly and the model can pass that on.
    """
    today = date.today()
    horizon = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    note = ""

    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        start = today
        note = f"start_date {start_date!r} was unreadable, used today instead. "

    days = max(1, min(int(days), MAX_FORECAST_DAYS))
    end = start + timedelta(days=days - 1)

    if start < today:
        note += f"Trip starts in the past ({start}); showing from today. "
        start = today
        end = min(start + timedelta(days=days - 1), horizon)

    if start > horizon:
        note += (
            f"Trip starts {start}, beyond the {MAX_FORECAST_DAYS}-day forecast "
            f"horizon. Showing the furthest available window instead; treat it "
            f"as a rough seasonal indication, not a forecast for the trip. "
        )
        end = horizon
        start = max(today, horizon - timedelta(days=days - 1))
    elif end > horizon:
        note += (
            f"Forecast only reaches {horizon}; the last "
            f"{(end - horizon).days} day(s) of the trip are not covered. "
        )
        end = horizon

    return start.isoformat(), end.isoformat(), note.strip()


def fetch_forecast(
    city: str,
    start_date: str,
    days: int,
    timeout: float = 20.0,
) -> Forecast:
    """Geocode then forecast. Raises WeatherError with a model-readable message."""
    with httpx.Client(timeout=timeout) as client:
        matches = geocode(city, client)
        if not matches:
            raise WeatherError(
                f"No place found matching {city!r}. Ask the user for a more "
                "specific city name, or a nearby larger city."
            )

        # Ambiguity is surfaced, not silently resolved — the model can ask.
        note_parts: list[str] = []
        if len(matches) > 1:
            others = ", ".join(
                f"{m.get('name')} ({m.get('country_code') or m.get('country') or '?'})"
                for m in matches[1:4]
            )
            note_parts.append(
                f"{city!r} also matched: {others}. If the user meant one of "
                "those, call check_weather again with a more specific name."
            )

        best = matches[0]
        start, end, window_note = _clamp_window(start_date, days)
        if window_note:
            note_parts.append(window_note)

        payload = _get(
            client,
            FORECAST_URL,
            {
                "latitude": best["latitude"],
                "longitude": best["longitude"],
                "daily": ",".join(
                    [
                        "temperature_2m_min",
                        "temperature_2m_max",
                        "precipitation_probability_max",
                        "precipitation_sum",
                        "wind_speed_10m_max",
                        "weather_code",
                    ]
                ),
                "timezone": "auto",
                "start_date": start,
                "end_date": end,
            },
        )

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        raise WeatherError(
            f"Weather service returned no daily data for {best.get('name')}. "
            "Tell the user and do not invent a forecast."
        )

    def col(key: str) -> list:
        values = daily.get(key) or []
        return list(values) + [None] * (len(dates) - len(values))

    tmin, tmax = col("temperature_2m_min"), col("temperature_2m_max")
    pprob, psum = col("precipitation_probability_max"), col("precipitation_sum")
    wind, code = col("wind_speed_10m_max"), col("weather_code")

    day_rows = [
        DayForecast(
            date=dates[i],
            temp_min_c=float(tmin[i] if tmin[i] is not None else 0.0),
            temp_max_c=float(tmax[i] if tmax[i] is not None else 0.0),
            precip_prob_pct=int(pprob[i] if pprob[i] is not None else 0),
            precip_mm=float(psum[i] if psum[i] is not None else 0.0),
            wind_max_kmh=float(wind[i] if wind[i] is not None else 0.0),
            condition=WMO.get(int(code[i]) if code[i] is not None else -1, "unknown"),
        )
        for i in range(len(dates))
    ]

    return Forecast(
        city_query=city,
        city_resolved=str(best.get("name") or city),
        country=str(best.get("country") or ""),
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
        timezone_name=str(payload.get("timezone") or ""),
        days=day_rows,
        note=" ".join(note_parts).strip(),
    )
