r"""
pyCICY.mathematica -- a small reader for Mathematica expression syntax.

The Oxford CICY distribution ships a Mathematica file alongside the plain
text one, and only the Mathematica file carries the freely acting discrete
symmetries classified by

    V. Braun, "On Free Quotients of Complete Intersection Calabi-Yau
    Manifolds", JHEP 04 (2011) 005, arXiv:1003.3235.

Rather than guess at the file's layout, this module parses Mathematica
expression syntax generically into Python objects and lets the caller inspect
what came back. That way an unexpected structure produces a readable report
instead of a wrong answer.

Supported syntax
----------------
* lists ``{a, b, c}``            -> ``list``
* rules ``a -> b``, ``a :> b``   -> :class:`Rule`
* assignments ``a = b``, ``a := b`` -> ``Expr("Set", ...)``
* function calls ``f[a, b]``     -> :class:`Expr`
* grouping ``(a + b)*c``
* parts ``expr[[i]]``            -> ``Expr("Part", ...)``
* integers, reals, rationals ``a/b``
* strings ``"..."``
* symbols ``Foo``, ``Global`had``
* comments ``(* ... *)`` are skipped

Not supported, and deliberately so: arithmetic evaluation, patterns,
replacement semantics, or anything else requiring a Mathematica kernel. This
reads data files; it is not an interpreter. Anything it cannot parse raises
:class:`MathematicaSyntaxError` with the offending offset, rather than
silently skipping input.
"""

__all__ = [
    "Expr", "Rule", "Symbol", "MathematicaSyntaxError",
    "loads", "load", "rules_to_dict", "describe",
]


class MathematicaSyntaxError(ValueError):
    """Raised when the input is not valid Mathematica expression syntax."""


class Symbol(str):
    """A bare Mathematica symbol, distinguished from a quoted string."""

    __slots__ = ()

    def __repr__(self):
        return "Symbol(%s)" % str.__repr__(self)


class Rule(object):
    """A Mathematica rule ``lhs -> rhs`` (or ``:>``)."""

    __slots__ = ("lhs", "rhs", "delayed")

    def __init__(self, lhs, rhs, delayed=False):
        self.lhs = lhs
        self.rhs = rhs
        self.delayed = delayed

    def __repr__(self):
        return "Rule(%r %s %r)" % (self.lhs, ":>" if self.delayed else "->",
                                   self.rhs)

    def __eq__(self, other):
        return (isinstance(other, Rule) and self.lhs == other.lhs
                and self.rhs == other.rhs and self.delayed == other.delayed)


class Expr(object):
    """A Mathematica function call ``head[args...]``."""

    __slots__ = ("head", "args")

    def __init__(self, head, args):
        self.head = head
        self.args = list(args)

    def __repr__(self):
        return "Expr(%s, %d args)" % (self.head, len(self.args))

    def __eq__(self, other):
        return (isinstance(other, Expr) and self.head == other.head
                and self.args == other.args)


# ------------------------------------------------------------------ parser

class _Parser(object):
    def __init__(self, text):
        self.s = text
        self.i = 0
        self.n = len(text)

    # -- lexing helpers

    def error(self, msg):
        line = self.s.count("\n", 0, self.i) + 1
        near = self.s[max(0, self.i - 40):self.i + 40].replace("\n", " ")
        raise MathematicaSyntaxError(
            "%s at offset %d (line %d), near: ...%s..."
            % (msg, self.i, line, near))

    def skip(self):
        while self.i < self.n:
            c = self.s[self.i]
            if c in " \t\r\n":
                self.i += 1
            elif self.s.startswith("(*", self.i):
                depth = 1
                self.i += 2
                while self.i < self.n and depth:
                    if self.s.startswith("(*", self.i):
                        depth += 1
                        self.i += 2
                    elif self.s.startswith("*)", self.i):
                        depth -= 1
                        self.i += 2
                    else:
                        self.i += 1
            elif c == "\\" and self.s.startswith("\\\n", self.i):
                self.i += 2          # line continuation
            else:
                return

    def peek(self):
        self.skip()
        return self.s[self.i] if self.i < self.n else ""

    # -- grammar

    def parse(self):
        value = self.parse_rule()
        self.skip()
        return value

    def parse_rule(self):
        left = self.parse_comparison()
        self.skip()
        if self.i < self.n and self.s[self.i] == "&":
            self.i += 1
            left = Expr("Function", [left])
            self.skip()
        if self.s.startswith("->", self.i):
            self.i += 2
            return Rule(left, self.parse_rule(), delayed=False)
        if self.s.startswith(":>", self.i):
            self.i += 2
            return Rule(left, self.parse_rule(), delayed=True)
        # Assignments. A data file typically wraps its table in one, e.g.
        # CICYlist = {...}; they are represented as Set / SetDelayed so the
        # right-hand side can be reached without special-casing the name.
        if self.s.startswith(":=", self.i):
            self.i += 2
            return Expr("SetDelayed", [left, self.parse_rule()])
        if (self.i < self.n and self.s[self.i] == "="
                and not self.s.startswith("==", self.i)
                and not self.s.startswith("=!", self.i)):
            self.i += 1
            return Expr("Set", [left, self.parse_rule()])
        return left

    # -- arithmetic
    #
    # The symmetry data records group actions as polynomials in the
    # homogeneous coordinates, e.g. x16^2*x2*x6, so the reader has to cope
    # with infix arithmetic. Nothing is evaluated: expressions are kept as
    # Plus / Times / Power / Minus trees, which is what a data reader wants.

    def parse_comparison(self):
        left = self.parse_plus()
        for op, head in (("==", "Equal"), ("!=", "Unequal"),
                         ("<=", "LessEqual"), (">=", "GreaterEqual")):
            self.skip()
            if self.s.startswith(op, self.i):
                self.i += len(op)
                return Expr(head, [left, self.parse_plus()])
        self.skip()
        if (self.i < self.n and self.s[self.i] in "<>"
                and not self.s.startswith("<|", self.i)):
            head = "Less" if self.s[self.i] == "<" else "Greater"
            self.i += 1
            return Expr(head, [left, self.parse_plus()])
        return left

    def parse_plus(self):
        left = self.parse_times()
        while True:
            self.skip()
            if self.i < self.n and self.s[self.i] == "+":
                self.i += 1
                left = Expr("Plus", [left, self.parse_times()])
            elif (self.i < self.n and self.s[self.i] == "-"
                  and not self.s.startswith("->", self.i)):
                self.i += 1
                left = Expr("Subtract", [left, self.parse_times()])
            else:
                return left

    def parse_times(self):
        left = self.parse_power()
        while True:
            self.skip()
            if (self.i < self.n and self.s[self.i] == "*"
                    and not self.s.startswith("*^", self.i)):
                self.i += 1
                left = Expr("Times", [left, self.parse_power()])
            elif (self.i < self.n and self.s[self.i] == "/"
                  and not self.s.startswith("/.", self.i)
                  and not self.s.startswith("/;", self.i)):
                self.i += 1
                left = Expr("Divide", [left, self.parse_power()])
            else:
                return left

    def parse_power(self):
        base = self.parse_unary()
        self.skip()
        if self.i < self.n and self.s[self.i] == "^":
            self.i += 1
            return Expr("Power", [base, self.parse_power()])   # right assoc
        return base

    def parse_unary(self):
        c = self.peek()
        if c == "-":
            # A negative literal stays a literal; anything else is Minus.
            j = self.i + 1
            while j < self.n and self.s[j] in " \t":
                j += 1
            if j < self.n and (self.s[j].isdigit() or self.s[j] == "."):
                return self.parse_atom()
            self.i += 1
            return Expr("Minus", [self.parse_unary()])
        if c == "+":
            self.i += 1
            return self.parse_unary()
        return self.parse_atom()

    def parse_atom(self):
        c = self.peek()
        if c == "":
            self.error("unexpected end of input")
        if c == "(":
            # A parenthesised subexpression. Note that "(*" is consumed as a
            # comment by skip() before we get here, so this is unambiguous.
            self.i += 1
            inner = self.parse_rule()
            if self.peek() != ")":
                self.error("expected ')'")
            self.i += 1
            return self.parse_postfix(inner)
        if c == "{":
            return self.parse_postfix(self.parse_list())
        if c == '"':
            return self.parse_string()
        if c == "-" or c == "+" or c.isdigit() or c == ".":
            return self.parse_number()
        if c == "#":
            self.i += 1
            start = self.i
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
            index = int(self.s[start:self.i]) if self.i > start else 1
            return self.parse_postfix(Expr("Slot", [index]))
        if c.isalpha() or c in "$`\\":
            return self.parse_postfix(self.parse_symbol_or_call())
        self.error("unexpected character %r" % c)

    def parse_postfix(self, value):
        """Handle trailing ``[[i]]`` parts and ``[...]`` applications."""
        while True:
            self.skip()
            if self.s.startswith("[[", self.i):
                self.i += 2
                args = [self.parse_rule()]
                while self.peek() == ",":
                    self.i += 1
                    args.append(self.parse_rule())
                if not self.s.startswith("]]", self.i):
                    self.error("expected ']]'")
                self.i += 2
                value = Expr("Part", [value] + args)
                continue
            if self.i < self.n and self.s[self.i] == "[":
                self.i += 1
                args = []
                if self.peek() == "]":
                    self.i += 1
                else:
                    while True:
                        args.append(self.parse_rule())
                        c = self.peek()
                        if c == ",":
                            self.i += 1
                            continue
                        if c == "]":
                            self.i += 1
                            break
                        self.error("expected ',' or ']'")
                value = Expr("Apply", [value] + args)
                continue
            return value

    def parse_list(self):
        self.i += 1                                  # consume {
        items = []
        if self.peek() == "}":
            self.i += 1
            return items
        while True:
            items.append(self.parse_rule())
            c = self.peek()
            if c == ",":
                self.i += 1
                continue
            if c == "}":
                self.i += 1
                return items
            self.error("expected ',' or '}' in list")

    def parse_string(self):
        self.i += 1
        out = []
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\\":
                self.i += 1
                if self.i >= self.n:
                    break
                esc = self.s[self.i]
                out.append({"n": "\n", "t": "\t", "r": "\r"}.get(esc, esc))
                self.i += 1
                continue
            if c == '"':
                self.i += 1
                return "".join(out)
            out.append(c)
            self.i += 1
        self.error("unterminated string")

    def parse_number(self):
        start = self.i
        if self.s[self.i] in "+-":
            self.i += 1
        digits = False
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
            digits = True
        is_real = False
        if self.i < self.n and self.s[self.i] == "." and not \
                self.s.startswith("..", self.i):
            is_real = True
            self.i += 1
            while self.i < self.n and self.s[self.i].isdigit():
                self.i += 1
                digits = True
        if not digits:
            self.error("malformed number")
        if self.s.startswith("*^", self.i):
            save = self.i
            self.i += 2
            if self.i < self.n and self.s[self.i] in "+-":
                self.i += 1
            if self.i < self.n and self.s[self.i].isdigit():
                while self.i < self.n and self.s[self.i].isdigit():
                    self.i += 1
                return float(self.s[start:self.i].replace("*^", "e"))
            self.i = save
        if self.i < self.n and self.s[self.i] in "eE":
            save = self.i
            self.i += 1
            if self.i < self.n and self.s[self.i] in "+-":
                self.i += 1
            if self.i < self.n and self.s[self.i].isdigit():
                is_real = True
                while self.i < self.n and self.s[self.i].isdigit():
                    self.i += 1
            else:
                self.i = save
        text = self.s[start:self.i]
        value = float(text) if is_real else int(text)

        # Mathematica writes exact rationals as a/b; only treat it as such
        # when both sides are integers, so that Times and Divide of symbols
        # are left alone.
        if not is_real:
            save = self.i
            self.skip()
            if self.i < self.n and self.s[self.i] == "/":
                self.i += 1
                self.skip()
                j = self.i
                if j < self.n and self.s[j].isdigit():
                    while self.i < self.n and self.s[self.i].isdigit():
                        self.i += 1
                    from fractions import Fraction
                    return Fraction(value, int(self.s[j:self.i]))
                self.i = save
            else:
                self.i = save
        return value

    def parse_symbol_or_call(self):
        start = self.i
        while self.i < self.n:
            c = self.s[self.i]
            if c.isalnum() or c in "$`":
                self.i += 1
            elif c == "\\" and self.s.startswith("\\[", self.i):
                end = self.s.find("]", self.i)
                if end < 0:
                    self.error("unterminated named character")
                self.i = end + 1
            else:
                break
        name = self.s[start:self.i]
        # "[[" is a Part, handled by parse_postfix, not a function call.
        if (self.i < self.n and self.s[self.i] == "["
                and not self.s.startswith("[[", self.i)):
            self.i += 1
            args = []
            if self.peek() == "]":
                self.i += 1
                return Expr(name, args)
            while True:
                args.append(self.parse_rule())
                c = self.peek()
                if c == ",":
                    self.i += 1
                    continue
                if c == "]":
                    self.i += 1
                    return Expr(name, args)
                self.error("expected ',' or ']' in %s[...]" % name)
        return Symbol(name)


def loads(text, all_expressions=False):
    """Parse Mathematica source text.

    Parameters
    ----------
    all_expressions : bool
        If True, parse a sequence of statements separated by ``;`` or
        newlines and return the list of them. Otherwise a single expression
        is expected and trailing input is an error.
    """
    parser = _Parser(text)
    if not all_expressions:
        value = parser.parse()
        parser.skip()
        if parser.i < parser.n:
            parser.error("trailing input after expression")
        return value

    out = []
    while True:
        parser.skip()
        while parser.i < parser.n and parser.s[parser.i] == ";":
            parser.i += 1
            parser.skip()
        if parser.i >= parser.n:
            return out
        out.append(parser.parse())


def load(path, **kwargs):
    """Parse a Mathematica file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return loads(fh.read(), **kwargs)


def rules_to_dict(value):
    """Turn a list of rules into a dict, leaving anything else alone."""
    if isinstance(value, Rule):
        return {value.lhs: value.rhs}
    if isinstance(value, list) and value and all(isinstance(v, Rule)
                                                 for v in value):
        return {v.lhs: v.rhs for v in value}
    return value


def describe(value, depth=0, max_depth=3, max_items=6):
    """A short structural summary, for working out an unfamiliar file.

    Prints shapes rather than contents, so it stays readable on a file with
    thousands of entries.
    """
    pad = "  " * depth
    if isinstance(value, list):
        lines = ["%slist of %d" % (pad, len(value))]
        if depth < max_depth:
            for item in value[:max_items]:
                lines.append(describe(item, depth + 1, max_depth, max_items))
            if len(value) > max_items:
                lines.append("%s  ... %d more" % (pad, len(value) - max_items))
        return "\n".join(lines)
    if isinstance(value, Rule):
        head = "%srule %r ->" % (pad, value.lhs)
        if depth < max_depth:
            return head + "\n" + describe(value.rhs, depth + 1, max_depth,
                                          max_items)
        return head + " ..."
    if isinstance(value, Expr):
        lines = ["%s%s[...] with %d args" % (pad, value.head, len(value.args))]
        if depth < max_depth:
            for item in value.args[:max_items]:
                lines.append(describe(item, depth + 1, max_depth, max_items))
        return "\n".join(lines)
    return "%s%s %r" % (pad, type(value).__name__, value)
