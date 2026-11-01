| time | who | prompt | what Codex did | what we changed by hand |
|---|---|---|---|---|
| 11:45 | Muskan | P1 stress tests | Wrote stress.py and tests/test_stress.py | Nothing changed by hand |
| 12:27 | Muskan | P2 adapter | Wrote adapters.py and check_baseline.py | Added BiLSTMAttention class to model_def.py by hand. Baseline 85.64% (328/383) matched the notebook |
| 12.54 | Muskan | P3 run_report | Wrote report.py and tests/test_report.py | Used --basetemp for pytest on Windows. Baseline 85.64% matched |
| 13:21 | Muskan | P4 Flask UI | Wrote app.py, templates/index.html, static files and tests/test_app.py | <jo haath se badla, nahi badla to "Nothing changed by hand">. Tested with my model and data, baseline 85.64% matched |