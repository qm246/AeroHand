"""macOS TCC permission checks (Accessibility, Camera, Microphone, Speech Recognition)."""

from __future__ import annotations

from typing import Dict, Tuple


def is_accessibility_trusted(prompt_user: bool = False) -> bool:
    """Return True if macOS Accessibility (AXIsProcessTrusted) is granted."""
    try:
        import ApplicationServices

        if prompt_user:
            opts = {ApplicationServices.kAXTrustedCheckOptionPrompt: True}
            return bool(ApplicationServices.AXIsProcessTrustedWithOptions(opts))
        return bool(ApplicationServices.AXIsProcessTrusted())
    except Exception:
        try:
            from Quartz import AXIsProcessTrustedWithOptions

            opts = {"AXTrustedCheckOptionPrompt": bool(prompt_user)}
            return bool(AXIsProcessTrustedWithOptions(opts))
        except Exception:
            return True


def is_camera_authorized() -> bool:
    """Check AVFoundation camera authorization status (3 = Authorized)."""
    try:
        import AVFoundation

        st = int(AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeVideo))
        return st == 3
    except Exception:
        return True


def is_microphone_authorized() -> bool:
    """Check AVFoundation microphone authorization status (3 = Authorized)."""
    try:
        import AVFoundation

        st = int(AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(AVFoundation.AVMediaTypeAudio))
        return st == 3
    except Exception:
        return False


def is_speech_authorized() -> bool:
    """Check macOS Speech Recognition authorization status (3 = Authorized)."""
    try:
        import Speech

        return int(Speech.SFSpeechRecognizer.authorizationStatus()) == 3
    except Exception:
        return False


def check_all_permissions() -> Dict[str, bool]:
    """Return a non-prompting snapshot of all macOS privacy permissions."""
    return {
        "accessibility": is_accessibility_trusted(prompt_user=False),
        "camera": is_camera_authorized(),
        "microphone": is_microphone_authorized(),
        "speech": is_speech_authorized(),
    }


def verify_permissions(dry_run: bool = False) -> Tuple[bool, str]:
    """Verify required runtime permissions and return `(ok, message)`."""
    if dry_run:
        return True, "Dry-run mode: OS event injection disabled."
    if not is_accessibility_trusted(prompt_user=True):
        return (
            False,
            "macOS Accessibility permission is required for live mouse/keyboard control.\n"
            "Enable your terminal/app in: System Settings -> Privacy & Security -> Accessibility\n"
            "Or run with `--dry-run` to test tracking safely without OS events.",
        )
    return True, "Permissions OK (Accessibility trusted; zero Input Monitoring required)."
