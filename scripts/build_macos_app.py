#!/usr/bin/env python3
"""
Build native macOS `/Users/macos/Desktop/AeroHand.app` bundle.

Architecture:
  - Generates a crisp Retina `AppIcon.icns` (16px..1024px) via Pillow + `iconutil`.
  - Compiles a tiny native arm64 Mach-O launcher (`Contents/MacOS/AeroHand`) embedding
    CPython 3.12 (`libpython3.12.dylib`) directly via `Py_InitializeFromConfig` so macOS
    LaunchServices, Dock icon, Activity Monitor, Camera TCC, and Accessibility TCC all
    identify the running process natively as `AeroHand.app` while loading live code from
    `/Users/macos/Documents/AeroHand/src` and `.venv`.
  - Ad-hoc signs the bundle with `codesign --force --deep --sign -`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_icon_png(out_png: Path, size: int = 1024) -> None:
    """Render a modern Airpoint-inspired dark-teal/blue gradient macOS app icon."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    margin = int(size * 0.08)
    radius = int(size * 0.22)

    # 1. Rounded squircle mask
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=radius, fill=255)

    # 2. Smooth diagonal gradient background (deep midnight -> ocean blue -> vibrant teal)
    grad = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    gpx = grad.load()
    span = max(1, size - 2 * margin)
    for y in range(margin, size - margin):
        ty = (y - margin) / span
        for x in range(margin, size - margin):
            tx = (x - margin) / span
            t = 0.55 * tx + 0.45 * (1.0 - ty)
            r = int(12 + 22 * t)
            g = int(22 + 165 * (t**1.15))
            b = int(46 + 185 * (t**0.85))
            gpx[x, y] = (r, g, b, 255)
    img.paste(grad, (0, 0), mask)

    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size // 2, size // 2

    # Subtle outer glass border
    draw.rounded_rectangle(
        [margin + 4, margin + 4, size - margin - 4, size - margin - 4],
        radius=radius - 2,
        outline=(255, 255, 255, 70),
        width=max(4, size // 180),
    )

    # Concentric touchless radar rings
    for mult, alpha in ((0.33, 55), (0.24, 95)):
        rr = int(size * mult)
        draw.ellipse(
            [cx - rr, cy - rr, cx + rr, cy + rr],
            outline=(180, 248, 240, alpha),
            width=max(4, size // 128),
        )

    # Stylized minimalist air-pointer chevron + glowing fingertip node
    s = size / 1024.0
    poly = [
        (int(365 * s), int(255 * s)),
        (int(365 * s), int(745 * s)),
        (int(490 * s), int(625 * s)),
        (int(580 * s), int(795 * s)),
        (int(655 * s), int(755 * s)),
        (int(565 * s), int(585 * s)),
        (int(725 * s), int(565 * s)),
    ]
    # Soft drop shadow
    shadow = [(x + int(10 * s), y + int(16 * s)) for x, y in poly]
    draw.polygon(shadow, fill=(4, 14, 28, 120))
    draw.polygon(poly, fill=(246, 252, 255, 248))

    # Glowing pinch dot near top-right of pointer
    dx, dy, dr = int(660 * s), int(325 * s), int(42 * s)
    draw.ellipse([dx - dr * 2, dy - dr * 2, dx + dr * 2, dy + dr * 2], fill=(45, 212, 191, 65))
    draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(94, 242, 224, 245))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    _ = px


def build_icns(base_png: Path, out_icns: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="aerohand_icon_") as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        base = Image.open(base_png)
        specs = [
            ("icon_16x16.png", 16),
            ("icon_16x16@2x.png", 32),
            ("icon_32x32.png", 32),
            ("icon_32x32@2x.png", 64),
            ("icon_128x128.png", 128),
            ("icon_128x128@2x.png", 256),
            ("icon_256x256.png", 256),
            ("icon_256x256@2x.png", 512),
            ("icon_512x512.png", 512),
            ("icon_512x512@2x.png", 1024),
        ]
        for fname, dim in specs:
            base.resize((dim, dim), Image.Resampling.LANCZOS).save(iconset / fname)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)], check=True)


def compile_embedded_launcher(out_binary: Path, repo_root: Path) -> None:
    venv_python = repo_root / ".venv" / "bin" / "python"
    base_prefix = Path(sys.base_prefix)
    include_dir = Path(sysconfig.get_path("include"))
    lib_dir = Path(sysconfig.get_config_var("LIBDIR"))
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    c_source = f"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {{
    chdir("{repo_root}");
    setenv("AEROHAND_BUNDLE_LAUNCH", "1", 1);

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;

    status = PyConfig_SetBytesString(&config, &config.executable, "{venv_python}");
    if (PyStatus_Exception(status)) goto fail;

    status = PyConfig_SetBytesString(&config, &config.home, "{base_prefix}");
    if (PyStatus_Exception(status)) goto fail;

    status = PyConfig_SetBytesString(&config, &config.run_module, "aerohand.ui.app");
    if (PyStatus_Exception(status)) goto fail;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) goto fail;
    PyConfig_Clear(&config);

    return Py_RunMain();

fail:
    PyConfig_Clear(&config);
    if (PyStatus_IsExit(status)) {{
        return status.exitcode;
    }}
    Py_ExitStatusException(status);
    return 1;
}}
"""
    with tempfile.TemporaryDirectory(prefix="aerohand_build_") as tmp:
        c_file = Path(tmp) / "launcher.c"
        c_file.write_text(c_source, encoding="utf-8")
        cmd = [
            "clang",
            "-O2",
            str(c_file),
            f"-I{include_dir}",
            f"-L{lib_dir}",
            f"-lpython{py_ver}",
            f"-Wl,-rpath,{lib_dir}",
            "-o",
            str(out_binary),
        ]
        subprocess.run(cmd, check=True)
    out_binary.chmod(0o755)


def write_info_plist(plist_path: Path) -> None:
    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>vi</string>
    <key>CFBundleDisplayName</key>
    <string>AeroHand</string>
    <key>CFBundleExecutable</key>
    <string>AeroHand</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.qm246.aerohand</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>AeroHand</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.0</string>
    <key>CFBundleVersion</key>
    <string>2</string>
    <key>CFBundleURLTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeRole</key>
            <string>Editor</string>
            <key>CFBundleURLName</key>
            <string>com.qm246.aerohand</string>
            <key>CFBundleURLSchemes</key>
            <array>
                <string>aerohand</string>
            </array>
        </dict>
    </array>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppleEventsUsageDescription</key>
    <string>AeroHand cần quyền Trợ năng (Accessibility) để điều khiển con trỏ, cuộn và cửa sổ.</string>
    <key>NSCameraUsageDescription</key>
    <string>AeroHand sử dụng webcam cục bộ 100% trên thiết bị để nhận diện bàn tay không chạm.</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>AeroHand cần quyền Microphone để ghi âm cho tính năng Đọc giọng nói thành văn bản.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>AeroHand sử dụng Apple Speech trên thiết bị để chuyển giọng nói Tiếng Việt và Tiếng Anh thành văn bản.</string>
</dict>
</plist>
"""
    plist_path.write_text(plist, encoding="utf-8")


def build_app(out_app: Path) -> Path:
    if out_app.exists():
        shutil.rmtree(out_app)
    macos_dir = out_app / "Contents" / "MacOS"
    res_dir = out_app / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    icon_png = REPO_ROOT / "assets" / "AppIcon.png"
    generate_icon_png(icon_png, size=1024)
    build_icns(icon_png, res_dir / "AppIcon.icns")
    write_info_plist(out_app / "Contents" / "Info.plist")
    compile_embedded_launcher(macos_dir / "AeroHand", REPO_ROOT)

    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(out_app)], check=True)
    return out_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AeroHand.app on Desktop")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path.home() / "Desktop" / "AeroHand.app"),
        help="Destination .app bundle path",
    )
    args = parser.parse_args()
    app_path = build_app(Path(args.out))
    print(f"[AeroHand] Built macOS app bundle -> {app_path}")


if __name__ == "__main__":
    main()
