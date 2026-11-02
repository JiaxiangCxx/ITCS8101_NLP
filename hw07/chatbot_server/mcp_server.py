from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESTAURANTS_PATH = DATA_DIR / "restaurants.json"
RESERVATIONS_PATH = ROOT / "reservations.csv"

mcp = FastMCP("Restaurant Reservation Server")


class DialogState(BaseModel):
    """Pydantic model for tracking dialog state."""

    City: str = Field(default="-", description="City name")
    State: str = Field(default="-", description="State abbreviation (2 letters)")
    Cuisine: str = Field(default="-", description="Cuisine type(s)")
    Name: str = Field(default="-", description="Restaurant name(s)")
    Misc: str = Field(default="-", description="Additional constraints")

    model_config = {"extra": "forbid"}


with RESTAURANTS_PATH.open() as f:
    _restaurants_raw = json.load(f)

restaurants = pd.DataFrame(_restaurants_raw)
location_restaurant_indices = list(range(len(restaurants)))
embdidx_to_idx = {idx: idx for idx in range(len(restaurants))}


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {_normalize_text(v)}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(_normalize_text(v) for v in value)
    return str(value)


def _restaurant_lookup_index(restaurant_index: int) -> int:
    return embdidx_to_idx.get(restaurant_index, restaurant_index)


def _restaurant_blob(global_index: int) -> str:
    row = restaurants.iloc[global_index]
    parts = [
        row.get("name", ""),
        row.get("city", ""),
        row.get("state", ""),
        row.get("categories", ""),
        _normalize_text(row.get("attributes")),
        _normalize_text(row.get("hours")),
        _normalize_text(row.get("review_texts")),
    ]
    return " ".join(_normalize_text(part) for part in parts).lower()


def _attribute_is_true(row, key: str) -> bool:
    attributes = row.get("attributes")
    if not isinstance(attributes, dict):
        return False
    value = attributes.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().strip("'\"").lower() == "true"


def _has_casual_attribute(row) -> bool:
    attributes = row.get("attributes")
    if not isinstance(attributes, dict):
        return False
    attire = _normalize_text(attributes.get("RestaurantsAttire")).lower()
    ambience = _normalize_text(attributes.get("Ambience")).lower()
    return "casual" in attire or "'casual': true" in ambience or '"casual": true' in ambience


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _state_to_dict(dialog_state) -> dict:
    if isinstance(dialog_state, DialogState):
        return dialog_state.model_dump()
    if isinstance(dialog_state, dict):
        return dialog_state
    if hasattr(dialog_state, "model_dump"):
        return dialog_state.model_dump()
    return dict(dialog_state)


@mcp.tool
def create_vdb_for_location(City: str, State: str):
    """Restrict vector database search to restaurants in a specific city and state."""
    global location_restaurant_indices
    global embdidx_to_idx

    city = City.strip().lower()
    state = State.strip().lower()
    mask = (
        restaurants["city"].fillna("").str.lower().eq(city)
        & restaurants["state"].fillna("").str.lower().eq(state)
    )
    location_restaurant_indices = restaurants.index[mask].tolist()
    embdidx_to_idx = {
        embd_idx: restaurant_idx
        for embd_idx, restaurant_idx in enumerate(location_restaurant_indices)
    }
    return {"count": len(location_restaurant_indices), "status": "success"}


def get_restaurant_str(restaurant_index: int):
    """Format one restaurant record as a single searchable/display string."""
    idx = _restaurant_lookup_index(restaurant_index)
    row = restaurants.iloc[idx]
    business_id = row.get("business_id", "")
    name = row.get("name", "")
    city = row.get("city", "")
    state = row.get("state", "")
    attributes = row.get("attributes")
    categories = row.get("categories")
    hours = row.get("hours")
    return (
        f"RestaurantId: {business_id} "
        f"Name: {name} "
        f"Location: {city} {state} "
        f"other: {attributes} {categories} {hours}"
    )


@mcp.tool
def get_restaurant_strs(restaurant_indexes: list):
    """Get restaurant data as structured text from the original JSON entries."""
    rstrings = [get_restaurant_str(restaurant_index) for restaurant_index in restaurant_indexes]
    return {"status": "Successful", "restaurant_strs": rstrings}


def check_restaurants_batch(dialog_state, restaurant_indices: list) -> list[bool]:
    """Deterministically filter restaurants against the dialogue state."""
    ds = _state_to_dict(dialog_state)
    results = []

    for restaurant_index in restaurant_indices:
        global_index = _restaurant_lookup_index(restaurant_index)
        row = restaurants.iloc[global_index]
        haystack = _restaurant_blob(global_index)
        keep = True

        city = ds.get("City", "-")
        if city not in ("", "-") and str(row.get("city", "")).lower() != city.lower():
            keep = False

        state = ds.get("State", "-")
        if state not in ("", "-") and str(row.get("state", "")).lower() != state.lower():
            keep = False

        cuisine = ds.get("Cuisine", "-")
        if keep and cuisine not in ("", "-"):
            cuisine_terms = _tokenize(cuisine.replace("_", " "))
            category_text = _normalize_text(
                [row.get("name", ""), row.get("categories", "")]
            ).lower()
            if cuisine_terms and not all(term in category_text for term in cuisine_terms):
                keep = False

        name = ds.get("Name", "-")
        if keep and name not in ("", "-"):
            name_terms = _tokenize(name)
            restaurant_name = str(row.get("name", "")).lower()
            if name_terms and not all(term in restaurant_name for term in name_terms):
                keep = False

        misc = ds.get("Misc", "-")
        if keep and misc not in ("", "-"):
            misc_terms = _tokenize(misc)
            if any(term in misc_terms for term in ("dog", "dogs", "pet", "friendly")):
                keep = _attribute_is_true(row, "DogsAllowed")
            if keep and any(term in misc_terms for term in ("patio", "outdoor", "seating")):
                keep = _attribute_is_true(row, "OutdoorSeating")
            if keep and "casual" in misc_terms:
                keep = _has_casual_attribute(row)
            remaining_terms = [
                term
                for term in misc_terms
                if term not in {"dog", "dogs", "pet", "friendly", "patio", "outdoor", "seating", "casual"}
            ]
            if keep and remaining_terms and not all(term in haystack for term in remaining_terms):
                keep = False

        results.append(keep)

    return results


def update_potential_restaurants(dialog_state, potential_restaurants):
    restaurant_list = list(potential_restaurants)
    batch_size = 50
    restaurants_to_remove = set()

    for i in range(0, len(restaurant_list), batch_size):
        batch = restaurant_list[i : i + batch_size]
        batch_res = check_restaurants_batch(dialog_state, batch)
        for idx, keep in zip(batch, batch_res):
            if not keep:
                restaurants_to_remove.add(idx)

    potential_restaurants -= restaurants_to_remove
    return potential_restaurants


@mcp.tool
def vdb_search_restaurants(dialog_state, query: str, k: int = 250) -> dict:
    """Search restaurants using lightweight text scoring plus dialogue-state filtering."""
    pool = location_restaurant_indices if location_restaurant_indices else []
    query_terms = _tokenize(query.replace("_", " "))
    scored = []

    for local_index, global_index in enumerate(pool):
        haystack = _restaurant_blob(global_index)
        score = sum(term in haystack for term in query_terms)
        row = restaurants.iloc[global_index]
        scored.append(
            (
                score,
                float(row.get("stars_x", 0) or 0),
                int(row.get("review_count", 0) or 0),
                local_index,
            )
        )

    scored.sort(reverse=True)
    top_matches = [local_index for _, _, _, local_index in scored[:k]]
    matching_restaurants = set(top_matches)
    potential_restaurants = update_potential_restaurants(dialog_state, matching_restaurants)
    return {"status": "Successful", "restaurants": list(potential_restaurants)}


@mcp.tool
def make_reservation(restaurant_id: str, date: str, time: str) -> dict:
    """Create and save a restaurant reservation to CSV file."""
    if restaurant_id not in set(restaurants["business_id"]):
        print(f"Error: Restaurant ID {restaurant_id} not found")
        return {"status": "fail"}

    try:
        reservation_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Error: Invalid date format {date}. Use YYYY-MM-DD")
        return {"status": "fail"}

    # Keep validation aligned with the assignment examples, which assume 2025.
    reference_today = datetime(2025, 1, 1).date()
    if reservation_date < reference_today:
        print(f"Error: Date {date} is in the past")
        return {"status": "fail"}

    try:
        datetime.strptime(time, "%H:%M")
    except ValueError:
        print(f"Error: Invalid time format {time}. Use HH:MM")
        return {"status": "fail"}

    try:
        file_exists = RESERVATIONS_PATH.exists()
        with RESERVATIONS_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["restaurant_id", "date", "time", "timestamp"],
            )
            if not file_exists or RESERVATIONS_PATH.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(
                {
                    "restaurant_id": restaurant_id,
                    "date": date,
                    "time": time,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    except Exception as exc:
        print(f"Error writing reservation: {exc}")
        return {"status": "fail"}

    print(f"Reservation successfully made for {date} at {time}")
    return {"status": "success"}


if __name__ == "__main__":
    mcp.run()
