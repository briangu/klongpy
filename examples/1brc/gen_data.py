#!/usr/bin/env python3
"""Generate test data for 1BRC challenge."""
import sys
import random

STATIONS = [
    "Hamburg", "Bulawayo", "Palembang", "St. John's", "Cracow",
    "Bridgetown", "Istanbul", "Roseau", "Conakry", "Istanbul",
    "Stockholm", "Mexico City", "Los Angeles", "Prague", "Tokyo",
    "Berlin", "Sydney", "Moscow", "Cairo", "Nairobi",
    "Lima", "Madrid", "Dubai", "Seoul", "Bangkok",
    "Singapore", "Helsinki", "Oslo", "Copenhagen", "Warsaw",
    "Budapest", "Bucharest", "Sofia", "Athens", "Rome",
    "Lisbon", "Amsterdam", "Brussels", "Zurich", "Vienna",
    "Mumbai", "Delhi", "Shanghai", "Beijing", "Guangzhou",
    "Ho Chi Minh City", "Jakarta", "Manila", "Taipei", "Kuala Lumpur",
]

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    fname = sys.argv[2] if len(sys.argv) > 2 else "measurements.txt"
    rng = random.Random(42)
    with open(fname, "w") as f:
        for _ in range(n):
            station = rng.choice(STATIONS)
            temp = rng.uniform(-20.0, 45.0)
            f.write(f"{station};{temp:.1f}\n")
    print(f"Generated {n} rows to {fname}")

if __name__ == "__main__":
    main()
