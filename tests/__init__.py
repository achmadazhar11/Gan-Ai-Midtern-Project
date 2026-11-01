"""Paket tes proyek.

File ini membuat `tests` menjadi paket biasa. Tanpa file ini, paket lain
bernama `tests` di site-packages (ada di Colab) menang saat import, sehingga
`from tests.helpers import ...` gagal dengan ModuleNotFoundError.
"""
