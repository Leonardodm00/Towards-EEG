"""render_previews.py -- static PNG previews of the repository's interactive HTML figures and its SVG (default state).

Run:  python3 render_previews.py <folder with the .html/.svg files> <output folder>
Needs Playwright with Chromium. The HTML files are self-contained (no network); each preview is the page as it
loads, before any slider is moved. The SVG is wrapped in a 1000 px wide page and screenshotted as an element.
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright


def main(src, out):
    src, out = pathlib.Path(src), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=2)
        for f in sorted(src.glob("*.html")):
            pg.goto(f.resolve().as_uri()); pg.wait_for_timeout(800)
            pg.screenshot(path=str(out / (f.stem + "_preview.png")), full_page=True)
            print(f.stem + "_preview.png")
        for f in sorted(src.glob("*.svg")):
            svg = f.read_text(encoding="utf-8").replace("<svg", "<svg style='width:1000px;height:auto'", 1)
            tmp = out / ("_" + f.stem + ".html")
            tmp.write_text("<html><body style='margin:0;background:#fff'><div style='width:1000px'>" + svg +
                           "</div></body></html>", encoding="utf-8")
            pg.goto(tmp.resolve().as_uri()); pg.wait_for_timeout(500)
            pg.locator("svg").first.screenshot(path=str(out / (f.stem + "_preview.png")), timeout=15000)
            tmp.unlink()
            print(f.stem + "_preview.png")
        b.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
