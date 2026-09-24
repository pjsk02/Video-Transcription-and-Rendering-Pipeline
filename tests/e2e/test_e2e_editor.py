"""E2E-05 — full clip editor: tabs (dirty dot), sliders mark dirty + update box,
scroll-zoom + drag-pan, magnifier cycle, trim drag + snap, playhead, caption
toggle/edit/validate, audio mute/volume, Apply no-op vs framing-change pill flip.

Apply is asserted at the pill-state level only; the real re-render is exercised by
test_e2e_real_render.py.
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import Page, expect

JOB_REAL = "e2eseed0001"

pytestmark = pytest.mark.e2e


def _open_editor(base_url: str, page: Page, idx: int = 2):
    page.goto(f"{base_url}/job/{JOB_REAL}/clip/{idx}/edit")
    expect(page.locator("#apply")).to_be_visible(timeout=10_000)


def test_editor_boots(base_url: str, page: Page):
    _open_editor(base_url, page)
    expect(page.locator("#asps .asp")).to_have_count(3)
    expect(page.locator("#pill")).to_have_text("Idle")
    expect(page.locator("#apply")).to_be_disabled()


def test_editor_slider_marks_dirty_and_updates_box(base_url: str, page: Page):
    _open_editor(base_url, page)
    box_before = page.eval_on_selector("#box", "el=>el.style.width")
    # Bump zoom → crop box shrinks, aspect tab gets a dirty dot, pill flips.
    page.eval_on_selector("#zoom", "el=>{el.value='1.6';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    expect(page.locator("#zv")).to_have_text("1.60×")
    box_after = page.eval_on_selector("#box", "el=>el.style.width")
    assert box_before != box_after
    expect(page.locator("#pill")).to_have_text("● Unsaved changes")
    # Active tab carries a dirty marker.
    assert page.locator('#asps .asp[data-a="9:16"] .dirty').count() == 1
    expect(page.locator("#apply")).to_be_enabled()


def test_editor_face_suggestion_previews_unsaved_9x16_crop(base_url: str, page: Page):
    payload = {"aspect": "9:16", "transform": {"zoom": 1.4, "x": 0.6, "y": 0.1},
               "faces_found": 8, "frames_sampled": 12, "cached": False}
    page.route("**/suggest-crop", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(payload)))
    _open_editor(base_url, page)
    page.locator('#asps .asp[data-a="1:1"]').click()
    page.locator("#facecrop").click()
    expect(page.locator("#zv")).to_have_text("1.40×")
    expect(page.locator('#asps .asp[data-a="9:16"]')).to_have_attribute("aria-pressed", "true")
    expect(page.locator("#facehint")).to_contain_text("8 of 12 sampled frames")
    expect(page.locator("#apply")).to_be_enabled()


def test_editor_tab_switch_keeps_per_aspect_state(base_url: str, page: Page):
    _open_editor(base_url, page)
    page.eval_on_selector("#zoom", "el=>{el.value='2';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    expect(page.locator("#zv")).to_have_text("2.00×")
    page.locator('#asps .asp[data-a="1:1"]').click()
    expect(page.locator("#zv")).to_have_text("1.00×")  # 1:1 untouched
    page.locator('#asps .asp[data-a="9:16"]').click()
    expect(page.locator("#zv")).to_have_text("2.00×")  # 9:16 retained


def test_editor_scroll_zoom_and_drag_pan(base_url: str, page: Page):
    _open_editor(base_url, page)
    page.locator("#of").hover()
    page.mouse.wheel(0, -400)
    expect(page.locator("#zv")).not_to_have_text("1.00×")
    # Drag to pan (zoomed in → has slack).
    x_before = page.eval_on_selector("#ox", "el=>parseFloat(el.value)")
    y_before = page.eval_on_selector("#oy", "el=>parseFloat(el.value)")
    box = page.locator("#of").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] / 2 - 70, box["y"] + box["height"] / 2 - 70, steps=6)
    page.mouse.up()
    x_after = page.eval_on_selector("#ox", "el=>parseFloat(el.value)")
    y_after = page.eval_on_selector("#oy", "el=>parseFloat(el.value)")
    assert (x_after != x_before) or (y_after != y_before)


def test_editor_magnifier_cycle(base_url: str, page: Page):
    _open_editor(base_url, page)
    btn = page.locator("#magbtn")
    expect(btn).to_have_text("🔍 Inspect 1×")
    btn.click()
    expect(btn).to_have_text("🔍 Inspect 1.5×")
    btn.click()
    expect(btn).to_have_text("🔍 Inspect 2×")
    btn.click()
    expect(btn).to_have_text("🔍 Inspect 1×")


def test_editor_trim_handle_drag_and_snap(base_url: str, page: Page):
    _open_editor(base_url, page)
    left_before = page.eval_on_selector("#hin", "el=>el.style.left")
    box = page.locator("#track").bounding_box()
    hin = page.locator("#hin").bounding_box()
    # The trim handler binds its pointermove/pointerup listeners to `document` on
    # pointerdown; Chromium's synthetic mouse stream doesn't drive that reliably in
    # headless, so dispatch the pointer sequence with explicit clientX/Y. This still
    # exercises the real onMove/snap/renderTrim logic in the page.
    cx0, cy0 = hin["x"] + hin["width"] / 2, hin["y"] + hin["height"] / 2
    target_x = box["x"] + box["width"] * 0.2
    page.eval_on_selector(
        "#hin",
        "(el,a)=>el.dispatchEvent(new PointerEvent('pointerdown',"
        "{bubbles:true,cancelable:true,button:0,pointerId:1,clientX:a.x,clientY:a.y}))",
        arg={"x": cx0, "y": cy0},
    )
    page.evaluate(
        "a=>document.dispatchEvent(new PointerEvent('pointermove',"
        "{bubbles:true,cancelable:true,pointerId:1,clientX:a.x,clientY:a.y}))",
        {"x": target_x, "y": cy0},
    )
    page.evaluate(
        "a=>document.dispatchEvent(new PointerEvent('pointerup',"
        "{bubbles:true,cancelable:true,pointerId:1,clientX:a.x,clientY:a.y}))",
        {"x": target_x, "y": cy0},
    )
    left_after = page.eval_on_selector("#hin", "el=>el.style.left")
    assert left_after != left_before
    # Trim marks the clip dirty.
    expect(page.locator("#pill")).to_have_text("● Unsaved changes")


def test_editor_playhead_via_track_click(base_url: str, page: Page):
    _open_editor(base_url, page)
    box = page.locator("#track").bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.6, box["y"] + box["height"] / 2)
    # Setting srcvid.currentTime moves the playhead time, even without decode.
    moved = page.eval_on_selector("#srcvid", "el=>el.currentTime > 0")
    assert moved


def test_editor_caption_toggle_and_edit_and_validate(base_url: str, page: Page):
    _open_editor(base_url, page)
    rows = page.locator("#caps .caprow")
    expect(rows.first).to_be_visible()

    # Edit a caption text → marks dirty.
    text_input = page.locator('#caps .caprow input[data-k="text"]').first
    text_input.fill("edited caption text")
    expect(page.locator("#pill")).to_have_text("● Unsaved changes")

    # Invalid time (negative) is blocked: row flagged, Apply disabled, inline error.
    start_input = page.locator('#caps .caprow input[data-k="start"]').first
    start_input.fill("-5")
    expect(start_input).to_have_class(re.compile(r"\bbad\b"))
    expect(page.locator("#apply")).to_be_disabled()
    expect(page.locator('#caps .caperr[data-err="0"]')).to_be_visible()

    # Fix it → error clears.
    start_input.fill("0")
    expect(start_input).not_to_have_class(re.compile(r"\bbad\b"))

    # Caption toggle off dims the list.
    page.uncheck("#capson")
    opacity = page.eval_on_selector("#caps", "el=>el.style.opacity")
    assert opacity == "0.4"


def test_editor_audio_mute_disables_volume(base_url: str, page: Page):
    _open_editor(base_url, page)
    expect(page.locator("#vol")).to_be_enabled()
    page.check("#mute")
    expect(page.locator("#vol")).to_be_disabled()
    expect(page.locator("#pill")).to_have_text("● Unsaved changes")
    page.uncheck("#mute")
    expect(page.locator("#vol")).to_be_enabled()


def test_editor_apply_noop_then_framing_change_flips_pill(base_url: str, page: Page):
    _open_editor(base_url, page)
    # No changes yet → Apply is a no-op (disabled).
    expect(page.locator("#apply")).to_be_disabled()
    expect(page.locator("#status")).to_have_text("No changes yet.")

    # A framing change enables Apply; clicking flips the flow pill to rendering/queued.
    page.eval_on_selector("#zoom", "el=>{el.value='1.4';el.dispatchEvent(new Event('input',{bubbles:true}))}")
    expect(page.locator("#apply")).to_be_enabled()
    page.click("#apply")
    expect(page.locator("#pill")).to_have_class(re.compile(r"\brendering\b"))
