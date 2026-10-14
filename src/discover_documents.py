
import requests
from bs4 import BeautifulSoup

URL = (
    "https://investor.nvidia.com/"
    "financial-info/annual-reports-and-proxies/default.aspx"
)

response = requests.get(
    URL,
    timeout=30,
    headers={"User-Agent": "Mozilla/5.0"},
)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

print("Website:", response.url)
print("HTTP status:", response.status_code)
print("\nPossible annual report / proxy document links:\n")

for link in soup.find_all("a", href=True):
    title = " ".join(link.get_text(" ", strip=True).split())
    href = link["href"]

    if any(word in (title + " " + href).lower()
           for word in ["annual", "proxy", ".pdf"]):
        if href.startswith("/"):
            href = "https://investor.nvidia.com" + href

        print(f"{title or '[No link title]'}")
        print(f"  {href}\n")
