# EchoMind — Merged Project Folder

This folder combines your two uploaded folders (`EchoMind_App` and `Echomind`)
into one place, with two real bugs in the app fixed and verified.

## How to run the app
```
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501

## What was wrong (found by actually running the app, not just reading code)

1. **Sidebar covered the page on phones.** Below ~640px width the sidebar
   stayed docked at a fixed 300px and squeezed the rest of the page into a
   thin, overlapping sliver instead of collapsing. Fixed with a media query
   that turns the sidebar into a proper slide-over drawer on small screens
   (`app.py`, `BASE_CSS`, new `@media (max-width: 640px)` block).

2. **The light/dark toggle button silently failed to click about half the
   time.** Streamlit's own (invisible) header bar sits on top of the page
   with `pointer-events:auto`, and it overlapped roughly the top half of the
   toggle button. A click aimed at the button's center could land on the
   hidden header instead and do nothing. Fixed by disabling pointer events
   on the empty header (`pointer-events:none`), since it has no visible
   controls left anyway (menu/toolbar/decoration are already hidden).

   Both fixes were verified with a headless browser: before the fix the
   toggle click timed out / had to be forced; after the fix it succeeds
   normally, and the mobile layout no longer overlaps.

3. **Path mismatch in `Echomind/app/APP_ui.ipynb`.** That notebook is a
   byte-for-byte copy of `app.py`'s code, but it expects `models/` and
   `assets/hero.png` next to itself — in the original folder those live one
   directory up, and there's no `assets/` folder at all, so running it in
   place would show "model files not found." Kept here for reference only
   (renamed `APP_ui_superseded_by_app_py.ipynb`) — use `app.py` at the root
   of this folder instead, it's the working, corrected version.

Everything else (navigation, the Analyze flow, Contact form, About/Tips
pages) was tested by simulating clicks and text entry and produced no
errors — no changes needed there.

## Folder layout
```
app.py                 fixed Streamlit app (run this)
requirements.txt
.streamlit/config.toml
assets/hero.png
models/                 bilstm_model.keras + tokenizer.pkl + label_encoder.pkl (used by app.py)
                         lstm_model.keras (baseline model, kept for reference)
checkpoint/             bilstm_best.keras (training checkpoint, reference only)
notebooks/              echomind.ipynb (training/EDA notebook)
docs/                   project reference doc + task brief
```

## Note on the raw datasets
`Echomind/data/` (Combined Data.csv, Suicide_Detection.csv, cleaned_dataset.csv,
combined_dataset.csv — about 550 MB total) was **not** copied into this
folder to keep the deliverable a reasonable size. They're untouched in your
original `Echomind` upload if you need them again for retraining.
