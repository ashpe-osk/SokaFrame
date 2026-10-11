"""Semantic class-name normalization for football object detectors."""


_CLASS_ALIASES = {
    "player": {"player", "players", "person"},
    "goalkeeper": {"goalkeeper", "goal keeper"},
    "referee": {"referee", "official"},
    "ball": {"ball", "football", "soccer ball"},
}


def normalize_class_name(name):
    """Normalize a detector label for matching against known class aliases."""
    return " ".join(str(name).strip().lower().replace("_", " ").replace("-", " ").split())


def map_class_categories(class_names):
    """Return detector class ID to semantic category mappings."""
    alias_to_category = {
        alias: category
        for category, aliases in _CLASS_ALIASES.items()
        for alias in aliases
    }
    entries = (
        class_names.items()
        if isinstance(class_names, dict)
        else enumerate(class_names)
    )
    return {
        int(class_id): alias_to_category.get(normalize_class_name(class_name), "other")
        for class_id, class_name in entries
    }
