"""Command-line interface for the weather dashboard."""

import argparse
import sys
from typing import Optional

from copilot_advanced_demo.config import Config
from copilot_advanced_demo.dashboard import WeatherDashboard


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Weather Dashboard - Get current weather and forecasts"
    )
    parser.add_argument(
        "city",
        help="City name (e.g., 'London' or 'London,UK')"
    )
    parser.add_argument(
        "--forecast", "-f",
        action="store_true",
        help="Show forecast instead of current weather"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=5,
        help="Number of days for forecast (1-7)"
    )
    parser.add_argument(
        "--analysis", "-a",
        action="store_true",
        help="Show weekly analysis"
    )
    parser.add_argument(
        "--api-key",
        help="Weather API key (or set WEATHER_API_KEY env var)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        if args.api_key:
            config = Config(api_key=args.api_key)
        else:
            try:
                config = Config.from_env()
            except ValueError:
                # Fall back to demo config
                config = Config(api_key="demo_key")
        
        dashboard = WeatherDashboard(config)

        # Execute requested action
        if args.analysis:
            print(dashboard.get_weekly_analysis(args.city))
        elif args.forecast:
            print(dashboard.display_forecast(args.city, args.days))
        else:
            print(dashboard.display_current_weather(args.city))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())