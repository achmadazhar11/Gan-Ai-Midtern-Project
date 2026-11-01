"""Paket skrip proyek.

File ini membuat `scripts` menjadi paket biasa, supaya `from scripts import ...`
di tes tidak tertimpa paket lain bernama `scripts` di site-packages.
Skrip tetap dijalankan langsung, misalnya `python scripts/measure_vram.py`.
"""
