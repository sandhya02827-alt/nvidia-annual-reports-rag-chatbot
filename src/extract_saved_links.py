
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin

html_file = Path("nvidia_reports_page.html")

if not html_file.exists():
    raise FileNotFoundError(
        "Saved webpage project folder mein nahi mili."
    )

html = html_file.read_text(encoding="utf-8", errors="replace")
soup = BeautifulSoup(html, "html.parser")

print("Saved HTML file:", html_file.resolve())
print("Page title:", soup.title.get_text(strip=True) if soup.title else "Not found")

links = []

for a in soup.find_all("a", href=True):
    title = " ".join(a.get_text(" ", strip=True).split())
    url = urljoin(
        "https://investor.nvidia.com/financial-info/annual-reports-and-proxies/default.aspx",
        a["href"],
    )

    if any(word in (title + " " + url).lower()
           for word in ["annual", "proxy", ".pdf"]):
        links.append((title or "[No title]", url))

# Remove duplicates while preserving order.
links = list(dict.fromkeys(links))

print(f"\nPossible document links found: {len(links)}\n")

for title, url in links:
    print(title)
    print(url)
    print("-" * 60)

if not links:
    print(
        "\nNo matching links found. The saved webpage may not "
        "contain the dynamically loaded document list."
    )
