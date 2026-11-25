"""macOS Camera & Accessibility permission helpers (adapted from GestureCtl)."""

from __future__ import annotations

import ctypes
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def is_accessibility_trusted(prompt_user: bool = False) -> bool:
    try:
        app_services = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        if prompt_user:
            cf = ctypes.cdll.LoadLibrary(
                "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
            )
            key = ctypes.c_void_p.in_dll(app_services, "kAXTrustedCheckOptionPrompt")
            val = ctypes.c_void_p.in_dll(cf, "kCFBooleanTrue")
            cf.CFDictionaryCreate.restype = ctypes.c_void_p
            cf.CFDictionaryCreate.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            opts = cf.CFDictionaryCreate(
                None, (ctypes.c_void_p * 1)(key), (ctypes.c_void_p * 1)(val), 1, None, None
            )
            app_services.AXIsProcessTrustedWithOptions.restype = ctypes.c_bool
            app_services.AXIsProcessTrustedWithOptions.argtypes = [ctypes.c_void_p]
            trusted = bool(app_services.AXIsProcessTrustedWithOptions(opts))
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            cf.CFRelease(opts)
            return trusted
        app_services.AXIsProcessTrusted.restype = ctypes.c_bool
        app_services.AXIsProcessTrusted.argtypes = []
        return bool(app_services.AXIsProcessTrusted())
    except Exception as exc:
        logger.debug(f"[PERM] Accessibility check failed: {exc}")
        return False


def verify_permissions(dry_run: bool = False) -> Tuple[bool, str]:
    if dry_run:
        return True, "Dry-run mode: OS input injection disabled."
    if not is_accessibility_trusted(prompt_user=True):
        return (
            False,
            "Accessibility permission required: System Settings -> Privacy & Security -> Accessibility "
            "-> enable Terminal / VS Code / AeroHand.",
        )
    return True, "Accessibility permission verified."
