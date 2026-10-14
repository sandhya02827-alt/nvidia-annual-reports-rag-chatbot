
import csv
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_FILE = PROJECT_ROOT / "nvidia_reports_page.html"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_FILE = RAW_DIR / "documents_manifest.csv"

PAGE_URL = (
    "https://investor.nvidia.com/"
    "financial-info/annual-reports-and-proxies/default.aspx"
)

def find_pdf_links():
    html = HTML_FILE.read_text(
        encoding="utf-8", errors="replace"
    )
    soup = BeautifulSoup(html, "html.parser")
    documents = []

    for link in soup.find_all("a", href=True):
        title = " ".join(link.get_text(" ", strip=True).split())
        url = urljoin(PAGE_URL, link["href"])

        if ".pdf" not in url.lower():
            continue

        text = (title + " " + url).lower()

        if "annual report" in text:
            doc_type = "annual_report"
        elif "proxy statement" in text:
            doc_type = "proxy_statement"
        else:
            continue

        match = re.search(r"\b(19\d{2}|20\d{2})\b", title)
        if not match:
            match = re.search(r"\b(19\d{2}|20\d{2})\b", url)

        if not match:
            continue

        documents.append({
            "year": int(match.group(1)),
            "type": doc_type,
            "title": title,
            "url": url,
        })

    unique = {}
    for doc in documents:
        unique[doc["url"]] = doc

    return list(unique.values())


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not HTML_FILE.exists():
        raise FileNotFoundError(
            f"Saved webpage not found: {HTML_FILE}"
        )

    documents = find_pdf_links()
    print(f"PDF links discovered: {len(documents)}")

    rows = []

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0"
        })

        for index, doc in enumerate(documents, start=1):
            filename = f'{doc["year"]}_{doc["type"]}.pdf'
            destination = RAW_DIR / filename
            temporary = RAW_DIR / (filename + ".part")

            print(
                f"\n[{index}/{len(documents)}] "
                f'{doc["year"]} {doc["type"]}'
            )

            # Skip only files that already look like valid PDFs.
            if destination.exists():
                with destination.open("rb") as file:
                    valid = file.read(5) == b"%PDF-"

                if valid:
                    print("Already downloaded; skipping.")
                    rows.append({
                        **doc,
                        "status": "downloaded",
                        "local_path": str(
                            destination.relative_to(PROJECT_ROOT)
                        ),
                        "error": "",
                    })
                    continue

                destination.unlink()

            try:
                with session.get(
                    doc["url"],
                    stream=True,
                    timeout=(20, 120),
                ) as response:
                    response.raise_for_status()

                    with temporary.open("wb") as file:
                        for chunk in response.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                file.write(chunk)

                with temporary.open("rb") as file:
                    valid_pdf = file.read(5) == b"%PDF-"

                if not valid_pdf:
                    raise ValueError(
                        "Downloaded file is not a valid PDF."
                    )

                temporary.replace(destination)

                size_mb = destination.stat().st_size / (
                    1024 * 1024
                )
                print(f"SUCCESS ({size_mb:.2f} MB)")

                status = "downloaded"
                local_path = str(
                    destination.relative_to(PROJECT_ROOT)
                )
                error = ""

            except Exception as exc:
                temporary.unlink(missing_ok=True)
                print(f"FAILED: {exc}")
                status = "failed"
                local_path = ""
                error = str(exc)

            rows.append({
                **doc,
                "status": status,
                "local_path": local_path,
                "error": error,
            })

    fieldnames = [
        "year", "type", "title", "url",
        "status", "local_path", "error",
    ]

    with MANIFEST_FILE.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(
        row["status"] == "downloaded" for row in rows
    )
    failed = sum(row["status"] == "failed" for row in rows)

    print("\n========== DOWNLOAD SUMMARY ==========")
    print(f"Documents listed: {len(rows)}")
    print(f"Downloaded/verified: {successful}")
    print(f"Failed: {failed}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
