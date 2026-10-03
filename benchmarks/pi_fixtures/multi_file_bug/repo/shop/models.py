from dataclasses import dataclass


@dataclass(frozen=True)
class LineItem:
    sku: str
    price: float
    quantity: int = 1

    @property
    def subtotal(self) -> float:
        return self.price * self.quantity
