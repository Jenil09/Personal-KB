"""Concrete implementations of what the domain describes.

Adapters point inward: they import from `domain`, never the reverse, and
nothing here is imported by a router. Only the composition root binds one.
"""
