# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

from typing import Any, Dict

import pytest
from pydantic import BaseModel

from pyagentspec.adapters._utils import (
    render_nested_json_template,
    render_nested_object_template,
    render_template,
)


class _Member(BaseModel):
    userId: str
    roles: Any = None


@pytest.mark.parametrize(
    "template, inputs, expected",
    [
        ("a", {}, "a"),
        ("{{a}}", {"a": 1}, "1"),
        ("{{ a}} {{b }}", {"a": 1, "b": 2}, "1 2"),
        ("{{ a} {b }}", {"a": 1, "b": 2}, "{{ a} {b }}"),
        ("{{ a}{}{b }}", {"a": 1, "b": 2}, "{{ a}{}{b }}"),
        ("{{ a a a a }}", {"a": 1}, "{{ a a a a }}"),
        ("{{a}}{{b}}{{a}}{{a}}", {"a": 1, "b": 2}, "1211"),
        ("{{ b{{a}} }}{{b1}}", {"a": 1, "b": 2, "b1": 3}, "{{ b1 }}3"),
        ("{{{{a}}}}", {"a": " b ", "b": 2}, "{{ b }}"),
        ("{{a}}{{b}}", {"a": "{{b}}", "b": 2}, "{{b}}2"),
        ("{{a}}{{b}}", {"b": 2, "a": "{{b}}"}, "{{b}}2"),
        ("{{a}}", {".*": "b"}, "{{a}}"),
        ("{{a}}", {"a|b": "c"}, "{{a}}"),
        ("{{a}}", {"[abc]": "b"}, "{{a}}"),
        ("{{a}}", {"b)": "b"}, "{{a}}"),
        (
            "Here is the equation: {{a}} plus {{b}} equals {{c}}",
            {"a": "{{", "b": "}}", "plus": "SECRET"},
            "Here is the equation: {{ plus }} equals {{c}}",
        ),
        ("{{a}}{{b}}", {"a": "{{sec", "b": "ret}}", "secret": "SECRET"}, "{{secret}}"),
    ],
)
def test_template_are_properly_rendered(
    template: str, inputs: Dict[str, Any], expected: str
) -> None:
    assert render_template(template, inputs) == expected


@pytest.mark.parametrize(
    "template, inputs, expected",
    [
        ("a", {}, "a"),
        ("{{a}}", {"a": 1}, "1"),
        ("{{ a}} {{b }}", {"a": 1, "b": 2}, "1 2"),
        ("{{ a} {b }}", {"a": 1, "b": 2}, "{{ a} {b }}"),
        ("{{ a}{}{b }}", {"a": 1, "b": 2}, "{{ a}{}{b }}"),
        ("{{ a a a a }}", {"a": 1}, "{{ a a a a }}"),
        ("{{a}}{{b}}{{a}}{{a}}", {"a": 1, "b": 2}, "1211"),
        ("{{ b{{a}} }}{{b1}}", {"a": 1, "b": 2, "b1": 3}, "{{ b1 }}3"),
        ("{{{{a}}}}", {"a": " b ", "b": 2}, "{{ b }}"),
        ("{{a}}{{b}}", {"a": "{{b}}", "b": 2}, "{{b}}2"),
        ({"{{a}}": "{{a}}{{b}}"}, {"a": "{{b}}", "b": 2}, {"{{b}}": "{{b}}2"}),
        (
            {"{{a}}": {"{{a}}{{b}}": {"{{b}}": "{{a}}"}}},
            {"a": "{{b}}", "b": 2},
            {"{{b}}": {"{{b}}2": {"2": "{{b}}"}}},
        ),
        (
            "Here is the equation: {{a}} plus {{b}} equals {{c}}",
            {"a": "{{", "b": "}}", "plus": "SECRET"},
            "Here is the equation: {{ plus }} equals {{c}}",
        ),
        ("{{a}}{{b}}", {"a": "{{sec", "b": "ret}}", "secret": "SECRET"}, "{{secret}}"),
        (
            [{"id_{{a}}": "v{{b}}"}, {"inner": {"k": "{{c}}"}}],
            {"a": 1, "b": 2, "c": 3},
            [{"id_1": "v2"}, {"inner": {"k": "3"}}],
        ),
        (
            {"outer": [{"name": "n{{x}}"}, ("t{{y}}", {"d": "v{{z}}"})]},
            {"x": "X", "y": "Y", "z": "Z"},
            {"outer": [{"name": "nX"}, ("tY", {"d": "vZ"})]},
        ),
        (
            ("p{{a}}", ["l{{b}}", {"k": "{{a}}"}], {"end": "{{b}}"}),
            {"a": "A", "b": "B"},
            ("pA", ["lB", {"k": "A"}], {"end": "B"}),
        ),
        ({"set": {"a", "s{{x}}"}}, {"x": "X"}, {"set": {"a", "sX"}}),
        (
            {"bytes": [b"b{{bb}}", {"k": b"{{bb2}}"}]},
            {"bb": "BB", "bb2": "B2"},
            {"bytes": ["bBB", {"k": "B2"}]},
        ),
        ({"{{a}}": [{"{{b}}": "v{{c}}"}]}, {"a": "A", "b": "B", "c": "C"}, {"A": [{"B": "vC"}]}),
        (
            {"l1": [{"l2": [{"l3": "x {{x}}"}]}], "k{{y}}": "v"},
            {"x": "X", "y": "Y"},
            {"l1": [{"l2": [{"l3": "x X"}]}], "kY": "v"},
        ),
        (
            ["pre {{p}}", {"mid": ["{{p}}", {"deep": "d{{d}}"}]}, "suf {{s}}"],
            {"p": "P", "d": "D", "s": "S"},
            ["pre P", {"mid": ["P", {"deep": "dD"}]}, "suf S"],
        ),
        ({("{{k}}", 1): "val {{v}}"}, {"k": "K", "v": "V"}, {("K", 1): "val V"}),
        (
            {"mix": [None, 0, "{{z}}", {"inner": (True, "{{z}}")}]},
            {"z": "Z"},
            {"mix": [None, 0, "Z", {"inner": (True, "Z")}]},
        ),
    ],
)
def test_json_template_are_properly_rendered(
    template: str, inputs: Dict[str, Any], expected: str
) -> None:
    assert render_nested_object_template(template, inputs) == expected


@pytest.mark.parametrize(
    "template, inputs, expected",
    [
        # A whole-placeholder value keeps its type (list/dict/number/bool/None),
        # so structured tool arguments survive into a JSON body.
        ("{{a}}", {"a": [1, 2]}, [1, 2]),
        ("{{a}}", {"a": None}, None),
        ("{{a}}", {"a": 5}, 5),
        ("{{a}}", {"a": True}, True),
        ("{{a}}", {"a": {"k": "v"}}, {"k": "v"}),
        # Strings stay strings; embedded placeholders still interpolate.
        ("{{a}}", {"a": "oneOnOne"}, "oneOnOne"),
        ("id-{{a}}", {"a": 5}, "id-5"),
        # Dict keys are always rendered as strings; only values keep their type.
        (
            {"input": {"members": "{{members}}", "topic": "{{topic}}", "type": "{{type}}"}},
            {
                "members": [_Member(userId="u1", roles=None)],
                "topic": None,
                "type": "group",
            },
            {
                "input": {
                    "members": [{"userId": "u1", "roles": None}],
                    "topic": None,
                    "type": "group",
                }
            },
        ),
    ],
)
def test_json_body_template_preserves_value_types(
    template: Any, inputs: Dict[str, Any], expected: Any
) -> None:
    assert render_nested_json_template(template, inputs) == expected
