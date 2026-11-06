import pytest

from tvd.registry import Registry


class Shape:
    pass


def shapes_with_a_square():
    shapes: Registry[Shape] = Registry("shape")

    @shapes.register("square")
    class Square(Shape):
        def __init__(self, side: float = 1.0) -> None:
            self.side = side

    return shapes, Square


def test_create_builds_the_registered_class_with_its_options():
    shapes, Square = shapes_with_a_square()
    square = shapes.create("square", side=3)
    assert isinstance(square, Square) and square.side == 3


def test_a_name_can_be_registered_only_once():
    shapes, _ = shapes_with_a_square()
    with pytest.raises(ValueError, match="already registered"):

        @shapes.register("square")
        class Other(Shape):
            pass


def test_an_unknown_name_lists_the_known_ones():
    shapes, _ = shapes_with_a_square()
    with pytest.raises(ValueError, match="Unknown shape 'circle'. Available: square"):
        shapes.create("circle")


def test_names_are_sorted():
    shapes, _ = shapes_with_a_square()
    shapes.register("circle")(type("Circle", (Shape,), {}))
    assert shapes.names() == ["circle", "square"]
