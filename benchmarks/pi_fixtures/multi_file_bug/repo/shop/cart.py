from shop.models import LineItem
from shop.pricing import apply_discount


class Cart:
    def __init__(self) -> None:
        self.items: list[LineItem] = []
        self.discount_code: str | None = None

    def add(self, sku: str, price: float, quantity: int = 1) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.items.append(LineItem(sku, price, quantity))

    def subtotal(self) -> float:
        return sum(item.subtotal for item in self.items[1:])

    def total(self) -> float:
        return apply_discount(self.subtotal(), self.discount_code)
