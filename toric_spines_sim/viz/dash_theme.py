"""Theme helpers for Dash visualization apps."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict


THEME_TOKENS: Dict[str, Dict[str, str]] = {
    "light": {
        "app_background": "#f6f8fb",
        "text": "#1f2937",
        "muted_text": "#6b7280",
        "card_background": "#ffffff",
        "card_border": "#e5e7eb",
        "card_shadow": "0 2px 10px rgba(0,0,0,0.05)",
        "table_border": "#e5e7eb",
        "danger_text": "#b91c1c",
        "code_background": "#eef2ff",
        "control_background": "#ffffff",
        "control_border": "#d1d5db",
        "control_text": "#111827",
        "control_placeholder": "#6b7280",
        "control_placeholder_dim": "#9ca3af",
        "control_hover": "#f3f4f6",
        "control_selected": "#dbeafe",
        "chip_background": "#e5e7eb",
        "chip_text": "#111827",
        "button_background": "#ffffff",
        "button_border": "#d1d5db",
        "button_text": "#1f2937",
        "button_hover": "#f3f4f6",
        "button_disabled_background": "#f3f4f6",
        "button_disabled_text": "#9ca3af",
        "metadata_diff_highlight": "#fff7ed",
    },
    "dark": {
        "app_background": "#0f172a",
        "text": "#e5e7eb",
        "muted_text": "#94a3b8",
        "card_background": "#111827",
        "card_border": "#374151",
        "card_shadow": "0 2px 10px rgba(0,0,0,0.35)",
        "table_border": "#374151",
        "danger_text": "#fca5a5",
        "code_background": "#1f2937",
        "control_background": "#111827",
        "control_border": "#4b5563",
        "control_text": "#e5e7eb",
        "control_placeholder": "#9ca3af",
        "control_placeholder_dim": "#6b7280",
        "control_hover": "#1f2937",
        "control_selected": "#334155",
        "chip_background": "#334155",
        "chip_text": "#e5e7eb",
        "button_background": "#1f2937",
        "button_border": "#4b5563",
        "button_text": "#e5e7eb",
        "button_hover": "#374151",
        "button_disabled_background": "#111827",
        "button_disabled_text": "#6b7280",
        "metadata_diff_highlight": "rgba(251, 191, 36, 0.14)",
    },
}


def get_theme_tokens(theme_name: str | None) -> Dict[str, str]:
    """Return theme token map for a given theme name."""
    normalized = (theme_name or "light").lower()
    if normalized not in THEME_TOKENS:
        normalized = "light"
    return deepcopy(THEME_TOKENS[normalized])


def plotly_template_for(theme_name: str | None) -> str:
    """Map theme name to Plotly template."""
    normalized = (theme_name or "light").lower()
    return "plotly_dark" if normalized == "dark" else "plotly_white"


def base_layout_styles(tokens: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Build common style blocks from tokens."""
    app_style = {
        "fontFamily": "Inter, Segoe UI, Roboto, sans-serif",
        "backgroundColor": tokens["app_background"],
        "color": tokens["text"],
        "padding": "20px",
        "minHeight": "100vh",
    }
    card_style = {
        "backgroundColor": tokens["card_background"],
        "border": f"1px solid {tokens['card_border']}",
        "borderRadius": "12px",
        "padding": "14px",
        "boxShadow": tokens["card_shadow"],
        "marginBottom": "14px",
    }
    card_title_style = {"marginTop": "0", "marginBottom": "10px", "fontWeight": "600"}
    helper_text_style = {"marginBottom": "10px", "color": tokens["muted_text"]}
    empty_state_style = {
        "color": tokens["danger_text"],
        "fontWeight": "bold",
        "marginBottom": "10px",
    }
    stepper_label_style = {
        "minWidth": "140px",
        "textAlign": "center",
        "fontSize": "13px",
        "color": tokens["muted_text"],
    }
    metadata_cell_key_style = {
        "fontWeight": "600",
        "padding": "6px 10px",
        "borderBottom": f"1px solid {tokens['table_border']}",
    }
    metadata_cell_value_style = {
        "padding": "6px 10px",
        "borderBottom": f"1px solid {tokens['table_border']}",
    }
    metadata_diff_cell_style = {
        "backgroundColor": tokens["metadata_diff_highlight"],
    }
    metadata_table_style = {
        "width": "100%",
        "borderCollapse": "collapse",
        "fontSize": "13px",
    }
    metadata_header_cell_style = {
        "textAlign": "left",
        "padding": "8px 10px",
        "borderBottom": f"1px solid {tokens['table_border']}",
        "color": tokens["text"],
    }
    run_id_label_style = {"fontWeight": "600", "color": tokens["text"]}
    code_style = {
        "backgroundColor": tokens["code_background"],
        "padding": "2px 6px",
        "borderRadius": "6px",
    }
    button_style = {
        "borderRadius": "8px",
        "padding": "6px 10px",
        "cursor": "pointer",
    }
    radio_label_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "marginRight": "14px",
        "color": tokens["text"],
    }
    section_label_style = {"fontWeight": "600", "color": tokens["text"]}

    return {
        "app_style": app_style,
        "card_style": card_style,
        "card_title_style": card_title_style,
        "helper_text_style": helper_text_style,
        "empty_state_style": empty_state_style,
        "stepper_label_style": stepper_label_style,
        "metadata_cell_key_style": metadata_cell_key_style,
        "metadata_cell_value_style": metadata_cell_value_style,
        "metadata_diff_cell_style": metadata_diff_cell_style,
        "metadata_table_style": metadata_table_style,
        "metadata_header_cell_style": metadata_header_cell_style,
        "run_id_label_style": run_id_label_style,
        "code_style": code_style,
        "button_style": button_style,
        "radio_label_style": radio_label_style,
        "section_label_style": section_label_style,
    }


def dropdown_theme_css(theme_name: str | None) -> str:
    """Return CSS rules for dcc.Dropdown internals by theme."""
    tokens = get_theme_tokens(theme_name)
    root_class = f".theme-{(theme_name or 'light').lower()}"
    return f"""
{root_class} .themed-dropdown .Select-control {{
  background-color: {tokens["control_background"]} !important;
  border: 1px solid {tokens["control_border"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select {{
  color: {tokens["control_text"]} !important;
}}
/* Dash versions using newer react-select class naming */
{root_class} .themed-dropdown [class$="-control"] {{
  background-color: {tokens["control_background"]} !important;
  border-color: {tokens["control_border"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-placeholder {{
  color: {tokens["control_placeholder"]} !important;
}}
{root_class} .themed-dropdown .Select--single > .Select-control .Select-value,
{root_class} .themed-dropdown .Select--single > .Select-control .Select-value-label,
{root_class} .themed-dropdown .has-value.Select--single > .Select-control .Select-value {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-singleValue"] {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-value {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-placeholder"] {{
  color: {tokens["control_placeholder"]} !important;
}}
{root_class} .themed-dropdown .Select-input > input {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown input {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-menu-outer {{
  background-color: {tokens["control_background"]} !important;
  border: 1px solid {tokens["control_border"]} !important;
}}
{root_class} .themed-dropdown .Select-menu {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-menu"],
{root_class} .themed-dropdown [class$="-menuList"] {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-option {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-option.is-focused {{
  background-color: {tokens["control_hover"]} !important;
}}
{root_class} .themed-dropdown .Select-option.is-selected {{
  background-color: {tokens["control_selected"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-option"] {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-option"][class*="is-focused"] {{
  background-color: {tokens["control_hover"]} !important;
}}
{root_class} .themed-dropdown [class$="-option"][class*="is-selected"] {{
  background-color: {tokens["control_selected"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-noresults {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_placeholder_dim"]} !important;
}}
{root_class} .themed-dropdown .Select--multi .Select-value {{
  background-color: {tokens["chip_background"]} !important;
  border: 1px solid {tokens["control_border"]} !important;
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown .Select--multi .Select-value-label {{
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-multiValue"] {{
  background-color: {tokens["chip_background"]} !important;
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-multiValueLabel"] {{
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-multiValueRemove"] {{
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-multiValueRemove"]:hover {{
  background-color: {tokens["control_hover"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select--multi .Select-value-icon {{
  border-right: 1px solid {tokens["control_border"]} !important;
  color: {tokens["chip_text"]} !important;
}}
{root_class} .themed-dropdown .Select--multi .Select-value-icon:hover {{
  background-color: {tokens["control_hover"]} !important;
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-arrow {{
  border-top-color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-indicatorSeparator"] {{
  background-color: {tokens["control_border"]} !important;
}}
{root_class} .themed-dropdown [class$="-dropdownIndicator"] {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown .Select-clear {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class$="-clearIndicator"] {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown svg {{
  fill: {tokens["control_text"]} !important;
  color: {tokens["control_text"]} !important;
}}
/* Fallback rules for Dash/react-select variants with generated class names. */
{root_class} .themed-dropdown [class*="control"],
{root_class} .themed-dropdown [class*="menu"],
{root_class} .themed-dropdown [class*="option"],
{root_class} .themed-dropdown [class*="singleValue"],
{root_class} .themed-dropdown [class*="placeholder"],
{root_class} .themed-dropdown [class*="multiValue"] {{
  background-color: {tokens["control_background"]} !important;
  color: {tokens["control_text"]} !important;
  border-color: {tokens["control_border"]} !important;
}}
{root_class} .themed-dropdown input,
{root_class} .themed-dropdown div,
{root_class} .themed-dropdown span {{
  color: {tokens["control_text"]} !important;
}}
{root_class} .themed-dropdown [class*="placeholder"] {{
  color: {tokens["control_placeholder"]} !important;
}}
{root_class} .themed-step-button {{
  background-color: {tokens["button_background"]} !important;
  border: 1px solid {tokens["button_border"]} !important;
  color: {tokens["button_text"]} !important;
}}
{root_class} .themed-step-button:hover {{
  background-color: {tokens["button_hover"]} !important;
}}
{root_class} .themed-step-button:disabled {{
  background-color: {tokens["button_disabled_background"]} !important;
  color: {tokens["button_disabled_text"]} !important;
  cursor: not-allowed !important;
}}
{root_class} .themed-section-label {{
  font-weight: 600 !important;
  color: {tokens["text"]} !important;
}}
{root_class} .themed-radio label,
{root_class} .themed-radio span {{
  color: {tokens["text"]} !important;
}}
{root_class},
{root_class} h1,
{root_class} h2,
{root_class} h3,
{root_class} h4,
{root_class} h5,
{root_class} h6,
{root_class} label,
{root_class} span,
{root_class} p,
{root_class} td,
{root_class} th {{
  color: {tokens["text"]};
}}
"""

