"""Load AmazonHistoryPrice dataset from GitHub raw files."""
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/TianXiaSJTU/AmazonPriceHistory/main/data/AmazonHistoryPrice/"
CATEGORIES = [
    "automotive", "baby-products", "beauty", "books", "electronics",
    "health-personal-care", "home-kitchen", "industrial-scientific",
    "movies-tv", "music", "other", "patio-lawn-garden", "pet-supplies",
    "software", "sports-outdoors", "tools-home-improvement",
]


def parse_price(price_str: str) -> float:
    """Parse '$349.98' -> 349.98"""
    return float(price_str.replace("$", "").replace(",", "").strip())


def load_dataset(cache_dir: str = "./data_cache", split_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """Download and parse all AmazonHistoryPrice JSON files.

    Returns dict with 'train' and 'test' lists of product dicts.
    Each product has:
        - title, description, category
        - list_price, cost (= lowest_price), budget (= 0.8 * list_price)
        - codename
        - mi (bool): mutual interest? budget > cost
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    all_products = []
    for cat in CATEGORIES:
        url = BASE_URL + f"{cat}.json"
        local_file = cache_path / f"{cat}.json"

        if not local_file.exists():
            print(f"Downloading {cat}.json ...")
            try:
                urllib.request.urlretrieve(url, local_file)
            except Exception as e:
                print(f"  Failed to download {cat}.json: {e}")
                continue

        with open(local_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        for idx, item in enumerate(items):
            try:
                list_price = parse_price(item.get("list_price", "0"))
                cost = parse_price(item.get("lowest_price", "0"))
                if list_price <= 0 or cost <= 0:
                    continue

                budget = round(list_price * 0.8, 2)

                product = {
                    "codename": f"{cat}_{idx}",
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "category": cat,
                    "list_price": list_price,
                    "cost": cost,
                    "budget": budget,
                    "mi": budget > cost,  # mutual interest
                }
                all_products.append(product)
            except Exception as e:
                print(f"  Skipping item in {cat}: {e}")
                continue

    # Split: 802 train / 128 test (paper's split)
    random.seed(split_seed)
    random.shuffle(all_products)
    train = all_products[:802] if len(all_products) >= 930 else all_products[: int(len(all_products) * 0.86)]
    test = all_products[len(train):len(train) + 128] if len(all_products) >= 930 else all_products[len(train):]

    mi_count = sum(1 for p in all_products if p["mi"])
    ci_count = len(all_products) - mi_count
    print(f"Loaded {len(all_products)} products: {len(train)} train, {len(test)} test")
    print(f"  MI (mutual interest): {mi_count}, CI (conflict): {ci_count}")

    return {"train": train, "test": test}


def format_inventory(product: Dict[str, Any]) -> str:
    """Format a single product as inventory list for seller."""
    lines = [
        f"Inventory List",
        f"- codename: {product['codename']}",
        f"  title: {product['title']}",
        f"  description: {product['description'][:200]}",
        f"  category: {product['category']}",
        f"  list_price: ${product['list_price']:.2f}",
    ]
    return "\n".join(lines)


def format_shopping_list(product: Dict[str, Any]) -> str:
    """Format a single product as shopping list for buyer."""
    lines = [
        f"Shopping List",
        f"- codename: {product['codename']}",
        f"  title: {product['title']}",
        f"  description: {product['description'][:200]}",
        f"  budget_limit: ${product['budget']:.2f}",
    ]
    return "\n".join(lines)


def format_seller_private(product: Dict[str, Any]) -> str:
    """Seller's private cost info."""
    return f"cost_price: ${product['cost']:.2f}"
