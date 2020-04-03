#!/usr/bin/env python3

import unittest

from ml.rl.workflow.tagged_union import TaggedUnion


class U(TaggedUnion):
    a: str
    b: int


class TestTaggedUnion(unittest.TestCase):
    def test_valid_field(self):
        u = U(a="foo")
        self.assertEqual(u.selected_field, "a")
        self.assertEqual(u.value, "foo")
        self.assertEqual(u.a, "foo")
        self.assertTrue("a" in u)
        self.assertFalse("b" in u)
        print(u)
        hash(u)
        u.b = 1
        self.assertEqual(u.selected_field, "b")
        self.assertEqual(u.value, 1)
        self.assertEqual(u.b, 1)
        self.assertFalse("a" in u)
        self.assertTrue("b" in u)

        with self.assertRaises(AttributeError):
            u.a
        
        with self.assertRaises(AttributeError):
            u.c

        with self.assertRaises(AttributeError):
            u.c = 2

    def test_invalid_field(self):
        with self.assertRaises(AttributeError):
            U(c=1)

    def test_too_few_args(self):
        with self.assertRaises(TypeError):
            U()

    def test_too_many_args(self):
        with self.assertRaises(TypeError):
            U(a="foo", b=1)
