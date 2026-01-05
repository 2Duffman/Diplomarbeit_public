"""

import argparse
import logging
import os
import sys

from pathlib import Path
from typing import Union

import requests

from scrapy.selector import Selector

BASE_DIR = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)


def download_bgg_dump(
    username: str,
    password: str,
    target_dir: Union[str, Path],
    force_overwrite: bool = False,
) -> None:
    #Download the latest BGG data dump.

    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Downloading latest BGG dump to <%s>", target_dir)

    login_url = "https://boardgamegeek.com/login/api/v1"
    html_url = "https://boardgamegeek.com/data_dumps/bg_ranks"

    with requests.Session() as session:
        credentials = {
            "credentials": {
                "username": username,
                "password": password,
            }
        }
        login_response = session.post(login_url, json=credentials)
        login_response.raise_for_status()

        html_response = session.get(html_url)
        html_response.raise_for_status()

        selector = Selector(text=html_response.text)

        for link in selector.css("#maincontent a[download]"):
            download_url = link.xpath("@href").get()
            file_name = link.xpath("@download").get()
            file_path = target_dir / file_name

            LOGGER.info("Downloading <%s> to <%s>", download_url, file_path)

            if file_path.exists():
                if not force_overwrite:
                    LOGGER.info("Skipping <%s> as it already exists.", file_path)
                    continue
                LOGGER.info("Overwriting <%s> as it already exists.", file_path)

            download_response = session.get(download_url)
            download_response.raise_for_status()

            with file_path.open("wb") as file:
                file.write(download_response.content)

    LOGGER.info("Done.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Download the latest BGG data dump.")
    parser.add_argument(
        "--out-dir",
        "-d",
        default=BASE_DIR / "feeds" / "bgg_dump",
        help="Output directory",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite existing files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Log level (repeat for more verbosity)",
    )

    return parser.parse_args()


def main():
    #Command line entry point.

    args = _parse_args()

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose > 0 else logging.INFO,
        format="%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)s] %(message)s",
    )

    LOGGER.info(args)

    username = os.getenv("BGG_USERNAME")
    password = os.getenv("BGG_PASSWORD")

    download_bgg_dump(
        username=username,
        password=password,
        target_dir=args.out_dir,
        force_overwrite=args.force,
    )


if __name__ == "__main__":
    main()
"""

#=====================
#=====================
import requests

# BGG Login API
login_url = "https://boardgamegeek.com/login/api/v1"
data_dump_url = "https://boardgamegeek.com/data_dumps/bg_ranks"  # Ensure this is the correct dump URL

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Replace with your actual credentials
username = "Duffman_2"
password = "Nk3?nxm3ULZ^JtB"

session = requests.Session()
data = {
    "credentials": {
        "username": username,
        "password": password
    }
}

# Perform Login
login_response = session.post(login_url, json=data, headers=headers)

if login_response.status_code == 204:
    print("✅ Successfully logged in!")
else:
    print(f"❌ Login failed! Status Code: {login_response.status_code}, Response: {login_response.text}")
    exit()

# Try accessing a page that requires login
profile_response = session.get("https://boardgamegeek.com/user/" + username)

if "Log in" in profile_response.text:
    print("❌ Session is not authenticated. The login might not be working correctly.")
    exit()
else:
    print("✅ Successfully authenticated session!")

# Now attempt to download the data dump
dump_response = session.get(data_dump_url)

if dump_response.status_code == 200:
    print("✅ Data dump download successful!")
    with open("bgg_dump.zip", "wb") as f:
        f.write(dump_response.content)
    print("📁 File saved as 'bgg_dump.zip'")
else:
    print(f"❌ Failed to download data dump! Status Code: {dump_response.status_code}, Response: {dump_response.text}")
