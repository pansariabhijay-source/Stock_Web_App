# Design System Strategy: High-End Institutional Intelligence

## 1. Overview & Creative North Star
The vision for this design system is **"The Digital Architect."** We are moving away from the cluttered, flashing dashboards of traditional retail trading and toward a curated, editorial experience that commands authority. This is a platform for high-conviction decision-making, not high-frequency distraction.

To achieve this, the system rejects "template" layouts—the rigid, boxed-in grids that define standard SaaS. Instead, we utilize **Intentional Asymmetry** and **Tonal Depth**. By allowing high-density data tables to sit adjacent to expansive, airy "Intelligence" briefs, we create a visual rhythm that feels bespoke and human-led. We prioritize "Forecasts" over "Predictions" and "Intelligence" over "AI," ensuring the UI reflects a tool built for professionals who value signal over noise.

---

## 2. Color & Surface Architecture
This system is built on a foundation of deep, ink-like shadows and vibrant, functional accents.

### The "No-Line" Rule
Standard 1px borders are strictly prohibited for sectioning. They create visual friction and "cheapen" the interface. Boundaries must be defined through:
- **Background Color Shifts:** Use `surface-container-low` for secondary modules sitting atop a `surface` background.
- **Negative Space:** Use the **Spacing Scale** (specifically `6` and `8` tokens) to create logical groupings without physical dividers.

### Surface Hierarchy & Nesting
Think of the UI as a series of physical layers. We use a "Nested Depth" approach:
1.  **Base Layer:** `surface` (#0f131c)
2.  **Navigation/Sidebar:** `surface-container-low` (#181b25)
3.  **Primary Content Cards:** `surface-container` (#1c1f29)
4.  **Popovers/Modals:** `surface-container-highest` (#31353f)

### The "Glass & Gradient" Rule
To add a "signature" feel, floating elements (like forecast tooltips) should use **Glassmorphism**. Apply `surface-variant` at 60% opacity with a `20px` backdrop-blur. 
For primary CTAs, use a subtle linear gradient from `primary` (#b8c3ff) to `primary-container` (#2d5bff) at a 135-degree angle. This adds a "jewel" quality to high-utility actions.

---

## 3. Typography
The system utilizes a dual-font approach to balance institutional authority with data precision.

- **Editorial Expression (Plus Jakarta Sans):** Used for `display`, `headline`, and `title` levels. Its geometric clarity and wider stance provide an "editorial" feel that breaks the monotony of data.
- **Data Utility (Inter):** Used for `body` and `label` levels. **Critical Requirement:** All numerical data must use `font-variant-numeric: tabular-nums`. This ensures that tickers, prices, and percentages align perfectly in vertical columns, maintaining the "Terminal" integrity.

### Hierarchy Tones
- **Headlines:** Use `headline-md` in `on-surface` for high-level market summaries.
- **Supporting Labels:** Use `label-sm` in `on-surface-variant` to de-emphasize metadata, allowing the core price data (`title-lg`) to lead the eye.

---

## 4. Elevation & Depth

### Tonal Layering
Avoid shadows for static cards. Instead, achieve lift by "stacking" the surface tiers. A `surface-container-highest` element placed within a `surface-container` section creates a natural, soft lift that feels integrated into the platform's architecture.

### Ambient Shadows
When an element must float (e.g., a dropdown or context menu), use an **Ambient Shadow**:
- **Color:** `surface-container-lowest` at 40% opacity.
- **Blur:** Large (`24px` to `40px`).
- **Spread:** Minimal.
This mimics natural light dispersion in a dark environment rather than a harsh black drop-shadow.

### The "Ghost Border" Fallback
If contrast is legally required for accessibility, use a **Ghost Border**: The `outline-variant` token at 15% opacity. This provides a "suggestion" of a boundary without interrupting the visual flow.

---

## 5. Components

### Primary Buttons
- **Style:** High-contrast `primary` background with `on-primary` text.
- **Rounding:** `md` (0.375rem / 6px) for a sophisticated, modern corner.
- **State:** On hover, transition to `primary-fixed-variant` with a subtle `2px` glow using the `primary` color at 20% opacity.

### The Forecast Chip
- **Context:** Used to display "Intelligence" scores.
- **Style:** Use `secondary` (#4edea3) for positive forecasts and `tertiary` (#ffb2b7) for negative. Use a "Soft Fill" (accent color at 10% opacity) with the text in the full-strength accent color. No border.

### Data Lists & Tables
- **Rule:** **Strictly forbid horizontal dividers.**
- **Separation:** Use a vertical `spacing-2` (0.7rem) gap between rows. On hover, the row background should shift to `surface-bright` (#353943) with an `xl` (0.75rem) corner radius to "cradle" the data.

### Market Inputs
- **Style:** Minimalist. Use `surface-container-highest` as the background. 
- **Active State:** The bottom border animates from 0% to 100% width using the `primary` sapphire blue. Avoid full-box outlines.

---

## 6. Do's and Don'ts

### Do
- **Do** prioritize "Breathing Room." High-end finance platforms often fail by overcrowding. Use `spacing-12` (4rem) between major content sections.
- **Do** use `secondary` (Emerald) and `tertiary` (Rose) exclusively for directional data (gains/losses). Never use them for decorative purposes.
- **Do** use `surface-dim` for "inactive" or "out-of-hours" market states to provide a clear psychological shift for the user.

### Don't
- **Don't** use pure black (#000000) or pure white (#ffffff). It causes eye strain in a "Terminal" environment. Stick to the `surface` and `on-surface` palettes.
- **Don't** use standard "AI" sparkles or robot icons. We communicate intelligence through superior data visualization and refined typography.
- **Don't** use 1px solid borders to separate sidebar items. Use `spacing-1` and background-color hover states instead.