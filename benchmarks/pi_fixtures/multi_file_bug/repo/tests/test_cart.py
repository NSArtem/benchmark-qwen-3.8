import unittest

from shop.cart import Cart


class CartTests(unittest.TestCase):
    def make_cart(self):
        cart = Cart()
        cart.add("apple", 0.5, 4)
        cart.add("bread", 3.0)
        cart.add("milk", 1.25, 2)
        return cart

    def test_subtotal(self):
        self.assertAlmostEqual(self.make_cart().subtotal(), 7.5)

    def test_total_with_discount(self):
        cart = self.make_cart()
        cart.discount_code = "SAVE10"
        self.assertAlmostEqual(cart.total(), 6.75)

    def test_unknown_code(self):
        cart = self.make_cart()
        cart.discount_code = "BOGUS"
        with self.assertRaises(ValueError):
            cart.total()

    def test_rejects_zero_quantity(self):
        with self.assertRaises(ValueError):
            Cart().add("x", 1.0, 0)


if __name__ == "__main__":
    unittest.main()
